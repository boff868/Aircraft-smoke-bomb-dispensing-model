import numpy as np
from tqdm import tqdm

# ==============================================================================
# 1. 常量与物理环境设置 (源于问题描述和Q2.py)
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

# 预计算导弹相关参数 (这些是固定的，可以在目标函数外部计算一次)
DIR_M1 = POS_FAKE_TARGET - POS_M1_INIT
VEL_M1_VECTOR = V_M1_SCALAR * DIR_M1 / np.linalg.norm(DIR_M1)
T_HIT_FAKE_TARGET = np.linalg.norm(POS_FAKE_TARGET - POS_M1_INIT) / V_M1_SCALAR


# ==============================================================================
# 2. 目标函数 (将Q2.py的模拟过程封装)
# ==============================================================================
def objective_function(strategy):
    """
    计算给定策略下的有效遮蔽时间。
    PSO会尝试最小化这个函数的返回值。

    Args:
        strategy (list or np.array): 一个包含四个决策变量的向量。
            - strategy[0]: v_fy1, 无人机飞行速度 (m/s)
            - strategy[1]: theta_deg, 无人机飞行方向角度 (度), 0度为x轴正方向
            - strategy[2]: t_drop, 投放时间 (s)
            - strategy[3]: t_delay, 引信延时 (s)

    Returns:
        float: 负的有效遮蔽总时长 (-effective_screening_time)。
    """
    v_fy1, theta_deg, t_drop, t_delay = strategy

    # --- 2.1. 根据策略计算关键物理量 ---
    theta_rad = np.radians(theta_deg)
    vel_fy1_vector = np.array([
        v_fy1 * np.cos(theta_rad),
        v_fy1 * np.sin(theta_rad),
        0  # 等高度飞行
    ])

    pos_drop = POS_FY1_INIT + vel_fy1_vector * t_drop

    # 烟幕弹平抛运动
    dx_grenade = vel_fy1_vector[0] * t_delay
    dy_grenade = vel_fy1_vector[1] * t_delay
    dz_grenade = -0.5 * G * t_delay ** 2
    pos_detonation = pos_drop + np.array([dx_grenade, dy_grenade, dz_grenade])

    t_detonation = t_drop + t_delay

    # 如果起爆时间晚于导弹命中，则此策略无效
    if t_detonation >= T_HIT_FAKE_TARGET:
        return 0  # 返回0意味着遮蔽时间为0，即-0

    # --- 2.2. 定义模拟中使用的辅助函数 (作用域内) ---
    def get_missile_pos(t):
        return POS_M1_INIT + VEL_M1_VECTOR * t

    def get_cloud_center_pos(t):
        if t < t_detonation:
            return None
        dt_after = t - t_detonation
        sink_distance = V_CLOUD_SINK * dt_after
        return pos_detonation - np.array([0, 0, sink_distance])

    def is_line_segment_blocked(p_missile, p_target, p_cloud):
        ap = p_missile - p_target
        ab = p_cloud - p_target
        ap_dot_ap = np.dot(ap, ap)
        if ap_dot_ap == 0: return False
        t = np.dot(ab, ap) / ap_dot_ap
        t = np.clip(t, 0, 1)
        closest_point = p_target + t * ap
        return np.linalg.norm(closest_point - p_cloud) <= R_CLOUD

    # 缓存轮廓点以提高效率
    if 'silhouette_points' not in objective_function.__dict__:
        points = []
        thetas = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        for theta in thetas:
            x = R_TARGET * np.cos(theta) + POS_TRUE_TARGET_BOTTOM_CENTER[0]
            y = POS_TRUE_TARGET_BOTTOM_CENTER[1]
            z_top = POS_TRUE_TARGET_BOTTOM_CENTER[2] + H_TARGET
            z_bottom = POS_TRUE_TARGET_BOTTOM_CENTER[2]
            points.append(np.array([x, y, z_top]))
            points.append(np.array([x, y, z_bottom]))
        objective_function.silhouette_points = np.array(points)

    silhouette_points = objective_function.silhouette_points

    def is_screening_effective(t):
        missile_pos = get_missile_pos(t)
        cloud_pos = get_cloud_center_pos(t)
        if cloud_pos is None:
            return False

        for target_point in silhouette_points:
            if not is_line_segment_blocked(missile_pos, target_point, cloud_pos):
                return False  # 只要有一个点没被挡住，就无效
        return True

    # --- 2.3. 执行模拟 ---
    t_start = t_detonation
    t_end = min(t_detonation + T_CLOUD_EFFECTIVE_DURATION, T_HIT_FAKE_TARGET)
    dt = 0.1  # 时间步长，可以调整以平衡精度和速度

    effective_screening_time = 0.0
    current_time = t_start
    while current_time <= t_end:
        if is_screening_effective(current_time):
            effective_screening_time += dt
        current_time += dt

    # PSO是最小化算法，所以返回负值
    return -effective_screening_time


# ==============================================================================
# 3. PSO 算法参数设置
# ==============================================================================
# 粒子群参数
N_PARTICLES = 50  # 粒子数量
MAX_ITERATIONS = 100  # 最大迭代次数

# 惯性权重和学习因子
W = 0.5  # 惯性权重
C1 = 1.5  # 个体学习因子
C2 = 1.5  # 社会学习因子

# 决策变量的边界 (v_fy1, theta_deg, t_drop, t_delay)
BOUNDS = [
    (70, 140),  # 无人机速度 v_fy1 (m/s)
    (0, 360),  # 飞行角度 theta_deg (度)
    (0.1, 50.0),  # 投放时间 t_drop (s) - 设定一个合理的上限
    (0.1, 20.0)  # 引信延时 t_delay (s) - 设定一个合理的上限
]
N_DIMENSIONS = len(BOUNDS)

# ==============================================================================
# 4. PSO 算法主逻辑
# ==============================================================================
if __name__ == "__main__":
    print("--- 开始使用粒子群算法 (PSO) 优化问题二 ---")

    # 初始化粒子位置和速度
    particles_pos = np.zeros((N_PARTICLES, N_DIMENSIONS))
    for i in range(N_DIMENSIONS):
        min_b, max_b = BOUNDS[i]
        particles_pos[:, i] = np.random.rand(N_PARTICLES) * (max_b - min_b) + min_b

    particles_vel = np.zeros((N_PARTICLES, N_DIMENSIONS))

    # 初始化个体最优和全局最优
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.full(N_PARTICLES, np.inf)

    gbest_pos = np.zeros(N_DIMENSIONS)
    gbest_fitness = np.inf

    # --- 迭代寻优 ---
    for i in tqdm(range(MAX_ITERATIONS), desc="PSO 优化中"):
        # 1. 计算每个粒子的适应度
        for j in range(N_PARTICLES):
            fitness = objective_function(particles_pos[j])

            # 2. 更新个体最优 pbest
            if fitness < pbest_fitness[j]:
                pbest_fitness[j] = fitness
                pbest_pos[j] = np.copy(particles_pos[j])

        # 3. 更新全局最优 gbest
        min_fitness_idx = np.argmin(pbest_fitness)
        if pbest_fitness[min_fitness_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[min_fitness_idx]
            gbest_pos = np.copy(pbest_pos[min_fitness_idx])

        # 4. 更新所有粒子的速度和位置
        for j in range(N_PARTICLES):
            r1, r2 = np.random.rand(2)

            # 更新速度
            cognitive_vel = C1 * r1 * (pbest_pos[j] - particles_pos[j])
            social_vel = C2 * r2 * (gbest_pos - particles_pos[j])
            particles_vel[j] = W * particles_vel[j] + cognitive_vel + social_vel

            # 更新位置
            particles_pos[j] += particles_vel[j]

            # 5. 处理边界条件 (将粒子拉回搜索空间)
            for k in range(N_DIMENSIONS):
                min_b, max_b = BOUNDS[k]
                particles_pos[j, k] = np.clip(particles_pos[j, k], min_b, max_b)

        if i % 10 == 0:
            print(f" 迭代{i + 1} / {MAX_ITERATIONS}: 当前最优遮蔽时长 = {-gbest_fitness: .4f}s            ")

            # ==============================================================================
            # 5. 结果输出
            # ==============================================================================
            print("      " + " = "*60)
            print("--- PSO 优化完成 ---")
            print(f"找到的最优有效遮蔽时长: {-gbest_fitness:.4f} 秒")
            print("对应的最优投放策略为:")
            print(f"  - 无人机飞行速度 (v_fy1): {gbest_pos[0]:.4f} m/s")
            print(f"  - 无人机飞行角度 (theta): {gbest_pos[1]:.4f} 度")
            print(f"  - 投放时间 (t_drop):       {gbest_pos[2]:.4f} s")
            print(f"  - 引信延时 (t_delay):     {gbest_pos[3]:.4f} s")
            print("=" * 60)

