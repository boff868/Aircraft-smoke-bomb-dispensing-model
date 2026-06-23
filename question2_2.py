import numpy as np
from tqdm import tqdm

# ==============================================================================
# 1. 常量与物理环境设置 (保持不变，确保模拟逻辑一致)
# ==============================================================================
G = 9.8  # 重力加速度 (m/s^2)

# 目标信息
POS_FAKE_TARGET = np.array([0, 0, 0])
POS_TRUE_TARGET_BOTTOM_CENTER = np.array([0, 200, 0])
R_TARGET = 7.0  # 真目标半径 (m)
H_TARGET = 10.0  # 真目标高度 (m)

# 导弹M1信息
POS_M1_INIT = np.array([20000, 0, 2000])
V_M1_SCALAR = 300.0

# 无人机FY1初始信息
POS_FY1_INIT = np.array([17800, 0, 1800])

# 烟幕云团参数
R_CLOUD = 10.0
V_CLOUD_SINK = 3.0
T_CLOUD_EFFECTIVE_DURATION = 20.0

# 预计算导弹相关参数 (固定值，外部计算一次)
DIR_M1 = POS_FAKE_TARGET - POS_M1_INIT
VEL_M1_VECTOR = V_M1_SCALAR * DIR_M1 / np.linalg.norm(DIR_M1)
T_HIT_FAKE_TARGET = np.linalg.norm(POS_FAKE_TARGET - POS_M1_INIT) / V_M1_SCALAR


# ==============================================================================
# 2. 目标函数 (保持不变，确保适应度计算准确)
# ==============================================================================
def objective_function(strategy):
    v_fy1, theta_deg, t_drop, t_delay = strategy

    # 计算无人机速度向量与关键位置
    theta_rad = np.radians(theta_deg)
    vel_fy1_vector = np.array([v_fy1 * np.cos(theta_rad), v_fy1 * np.sin(theta_rad), 0])
    pos_drop = POS_FY1_INIT + vel_fy1_vector * t_drop

    # 烟幕弹平抛运动与起爆时间
    dx_grenade = vel_fy1_vector[0] * t_delay
    dy_grenade = vel_fy1_vector[1] * t_delay
    dz_grenade = -0.5 * G * t_delay ** 2
    pos_detonation = pos_drop + np.array([dx_grenade, dy_grenade, dz_grenade])
    t_detonation = t_drop + t_delay

    # 起爆晚于导弹命中则无效
    if t_detonation >= T_HIT_FAKE_TARGET:
        return 0

    # 辅助函数：导弹位置、烟幕位置、遮挡判断
    def get_missile_pos(t):
        return POS_M1_INIT + VEL_M1_VECTOR * t

    def get_cloud_center_pos(t):
        if t < t_detonation:
            return None
        return pos_detonation - np.array([0, 0, V_CLOUD_SINK * (t - t_detonation)])

    def is_line_segment_blocked(p_missile, p_target, p_cloud):
        ap = p_missile - p_target
        ab = p_cloud - p_target
        ap_dot_ap = np.dot(ap, ap)
        if ap_dot_ap == 0:
            return False
        t = np.clip(np.dot(ab, ap) / ap_dot_ap, 0, 1)
        closest_point = p_target + t * ap
        return np.linalg.norm(closest_point - p_cloud) <= R_CLOUD

    # 缓存真目标轮廓点（避免重复生成）
    if 'silhouette_points' not in objective_function.__dict__:
        points = []
        for theta in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            x = R_TARGET * np.cos(theta) + POS_TRUE_TARGET_BOTTOM_CENTER[0]
            y = POS_TRUE_TARGET_BOTTOM_CENTER[1]+R_TARGET * np.sin(theta)
            points.append(np.array([x, y, POS_TRUE_TARGET_BOTTOM_CENTER[2] + H_TARGET]))
            points.append(np.array([x, y, POS_TRUE_TARGET_BOTTOM_CENTER[2]]))
        objective_function.silhouette_points = np.array(points)

    # 遮蔽有效性判断
    def is_screening_effective(t):
        missile_pos = get_missile_pos(t)
        cloud_pos = get_cloud_center_pos(t)
        if cloud_pos is None:
            return False
        for p in objective_function.silhouette_points:
            if not is_line_segment_blocked(missile_pos, p, cloud_pos):
                return False
        return True

    # 统计有效遮蔽时间
    t_start, t_end = t_detonation, min(t_detonation + 20, T_HIT_FAKE_TARGET)
    effective_time = 0.0
    current_time = t_start
    while current_time <= t_end:
        if is_screening_effective(current_time):
            effective_time += 0.1
        current_time += 0.1

    return -effective_time  # PSO最小化，返回负值


# ==============================================================================
# 3. 优化后的PSO参数设置 (核心改动1：参数适配优化)
# ==============================================================================
N_PARTICLES = 80  # 粒子数从50增至80，扩大搜索范围
MAX_ITERATIONS = 150  # 迭代次数从100增至150，给足探索时间

# 动态惯性权重参数（替代固定W=0.5）
W_MAX = 0.9  # 初始大权重，增强全局探索
W_MIN = 0.4  # 后期小权重，增强局部利用

# 学习因子微调（增强社会学习，引导粒子向全局最优靠拢）
C1 = 1.4  # 个体学习因子（略降，减少局部固守）
C2 = 1.6  # 社会学习因子（略升，增强全局引导）

# 决策变量边界（保持不变，确保符合物理意义）
BOUNDS = [(70, 140), (0, 360), (0.1, 50.0), (0.1, 20.0)]
N_DIMENSIONS = len(BOUNDS)

# 位置扰动参数（核心改动2：跳出局部最优的关键）
PERTURB_PROB = 0.05  # 5%概率对粒子位置进行扰动
PERTURB_SCALE = 0.1  # 扰动幅度：边界范围的10%（平衡探索与稳定）


# ==============================================================================
# 4. 优化后的PSO算法主逻辑 (核心改动集中区)
# ==============================================================================
if __name__ == "__main__":
    print("--- 开始优化版粒子群算法 (PSO) 优化问题二 ---")

    # --------------------------
    # 优化1：拉丁超立方初始化 (替代随机初始化)
    # 作用：确保粒子在搜索空间均匀分布，避免初始聚集
    # --------------------------
    particles_pos = np.zeros((N_PARTICLES, N_DIMENSIONS))
    for i in range(N_DIMENSIONS):
        min_b, max_b = BOUNDS[i]
        # 拉丁超立方：将每个维度分成N_PARTICLES段，每段取1个点，确保均匀
        segments = np.linspace(min_b, max_b, N_PARTICLES + 1)
        for j in range(N_PARTICLES):
            particles_pos[j, i] = np.random.uniform(segments[j], segments[j + 1])

    # 速度初始化（保持不变，范围为对应维度边界的10%）
    particles_vel = np.zeros((N_PARTICLES, N_DIMENSIONS))
    for i in range(N_DIMENSIONS):
        min_b, max_b = BOUNDS[i]
        particles_vel[:, i] = np.random.uniform(-0.1 * (max_b - min_b), 0.1 * (max_b - min_b), N_PARTICLES)

    # 初始化个体最优和全局最优（保持不变）
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([objective_function(pos) for pos in particles_pos])
    gbest_idx = np.argmin(pbest_fitness)
    gbest_pos = np.copy(pbest_pos[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]

    # --------------------------
    # 迭代寻优（核心优化集中区）
    # --------------------------
    for iter in tqdm(range(MAX_ITERATIONS), desc="PSO 优化中"):
        # 优化2：动态惯性权重（随迭代次数线性递减）
        # 作用：前期大权重→全局探索，后期小权重→局部收敛
        W = W_MAX - (W_MAX - W_MIN) * (iter / MAX_ITERATIONS)

        # 1. 更新每个粒子的速度和位置
        for j in range(N_PARTICLES):
            # 随机因子（保持不变）
            r1, r2 = np.random.rand(2)

            # 速度更新公式（保持原框架，权重W动态变化）
            cognitive_vel = C1 * r1 * (pbest_pos[j] - particles_pos[j])
            social_vel = C2 * r2 * (gbest_pos - particles_pos[j])
            particles_vel[j] = W * particles_vel[j] + cognitive_vel + social_vel

            # 位置更新（保持不变）
            particles_pos[j] += particles_vel[j]

            # 优化3：位置边界处理（增加“反弹”逻辑，替代单纯截断）
            # 作用：避免粒子在边界“堆积”，增强边界区域探索
            for k in range(N_DIMENSIONS):
                min_b, max_b = BOUNDS[k]
                if particles_pos[j, k] < min_b:
                    particles_pos[j, k] = 2 * min_b - particles_pos[j, k]  # 反弹回边界内
                    particles_vel[j, k] *= -0.5  # 速度反向且减半，避免反复反弹
                elif particles_pos[j, k] > max_b:
                    particles_pos[j, k] = 2 * max_b - particles_pos[j, k]
                    particles_vel[j, k] *= -0.5

            # 优化4：随机位置扰动（迭代后期触发，跳出局部最优）
            # 作用：迭代80%后，若粒子未更新个体最优，强制扰动位置
            if iter > 0.8 * MAX_ITERATIONS:  # 后期触发（避免前期打乱探索）
                current_fitness = objective_function(particles_pos[j])
                if current_fitness >= pbest_fitness[j] + 1e-6:  # 粒子未改进
                    if np.random.rand() < PERTURB_PROB:  # 5%概率扰动
                        for k in range(N_DIMENSIONS):
                            min_b, max_b = BOUNDS[k]
                            # 扰动幅度：边界范围的10%，随机增减
                            particles_pos[j, k] += np.random.uniform(-PERTURB_SCALE*(max_b-min_b),
                                                                   PERTURB_SCALE*(max_b-min_b))

        # 2. 更新个体最优和全局最优（保持原逻辑，增加精度判断）
        for j in range(N_PARTICLES):
            current_fitness = objective_function(particles_pos[j])
            # 增加1e-6精度阈值，避免微小波动反复更新
            if current_fitness < pbest_fitness[j] - 1e-6:
                pbest_fitness[j] = current_fitness
                pbest_pos[j] = np.copy(particles_pos[j])

        # 更新全局最优
        current_gbest_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_gbest_idx] < gbest_fitness - 1e-6:
            gbest_fitness = pbest_fitness[current_gbest_idx]
            gbest_pos = np.copy(pbest_pos[current_gbest_idx])

        # 3. 结果输出（优化输出频率，每20次迭代输出一次，避免干扰）
        if (iter + 1) % 20 == 0 or iter == 0:
            print(f"\n迭代{iter + 1:3d}/{MAX_ITERATIONS}: 当前最优遮蔽时长 = {-gbest_fitness:6.4f}s")

    # ==============================================================================
    # 5. 最终结果输出（保持不变，补充优化信息）
    # ==============================================================================
    print("\n" + "=" * 60)
    print("--- 优化版PSO 优化完成 ---")
    print(f"优化策略：动态惯性权重 + 位置扰动 + 拉丁超立方初始化")
    print(f"最优有效遮蔽时长: {-gbest_fitness:.4f} 秒")
    print("对应的最优投放策略:")
    print(f"  - 无人机飞行速度 (v_fy1): {gbest_pos[0]:.2f} m/s")
    print(f"  - 无人机飞行角度 (theta): {gbest_pos[1]:.2f} 度")
    print(f"  - 投放时间 (t_drop):       {gbest_pos[2]:.2f} s")
    print(f"  - 引信延时 (t_delay):     {gbest_pos[3]:.2f} s")
    print("=" * 60)