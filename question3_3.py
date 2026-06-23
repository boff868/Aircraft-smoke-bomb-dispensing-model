import numpy as np
from tqdm import tqdm

# ==============================================================================
# 1. 常量与环境参数（严格遵循A题(3).pdf定义）
# ==============================================================================
G = 9.8  # 重力加速度(m/s²)

# 目标参数（A题(3).pdf：真目标为半径7m、高10m圆柱体）
POS_true_target = np.array([0, 200, 5])  # 真目标几何中心
r_target = 7.0  # 真目标半径(m)
h_target = 10.0  # 真目标高度(m)

# 武器与平台参数（A题(3).pdf初始位置与性能）
pos_m1_init = np.array([20000, 0, 2000])  # 导弹M1初始位置
v_m1 = 300.0  # 导弹飞行速度(m/s)，直指假目标（原点）
pos_fy1_init = np.array([17800, 0, 1800])  # 无人机FY1初始位置
v_fy1_min, v_fy1_max = 70.0, 140.0  # 无人机速度范围（A题(3).pdf明确）

# 烟幕参数（A题(3).pdf有效遮蔽规则）
r_cloud = 10.0  # 云团中心10m范围内有效
v_cloud_sink = 3.0  # 云团匀速下沉速度(m/s)
t_cloud_max = 20.0  # 起爆后20s内有效
min_drop_interval = 1.0  # 每架无人机投放两枚弹至少间隔1s（A题(3).pdf要求）

# 导弹轨迹预计算（A题(3).pdf推导：假目标为原点，导弹沿直线飞向假目标）
dir_m1 = -pos_m1_init  # 导弹飞行方向向量（从M1指向假目标）
vel_m1_vector = v_m1 * dir_m1 / np.linalg.norm(dir_m1)  # 导弹速度向量
t_m1_fly = np.linalg.norm(dir_m1) / v_m1  # 导弹命中假目标总时间（≈67.33s）


# ==============================================================================
# 2. 真目标关键点采样（A题(3).pdf几何特征，37个点覆盖核心区域）
# ==============================================================================
def generate_target_key_points():
    """生成真目标关键采样点，确保覆盖A题(3).pdf中圆柱体的所有易暴露区域"""
    points = []
    # 1. 真目标中心（1个点，避免核心区域漏遮）
    points.append(POS_true_target)
    # 2. 下底面圆周（12个点，每30°1个，z=0，A题(3).pdf下底面高度）
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = r_target * np.cos(theta)
        y = r_target * np.sin(theta) + 200  # 真目标y轴基准为200
        points.append(np.array([x, y, 0]))
    # 3. 上底面圆周（12个点，z=10，A题(3).pdf上底面高度）
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = r_target * np.cos(theta)
        y = r_target * np.sin(theta) + 200
        points.append(np.array([x, y, h_target]))
    # 4. 侧面中间（12个点，z=5，避免侧面边缘漏遮）
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = r_target * np.cos(theta)
        y = r_target * np.sin(theta) + 200
        points.append(np.array([x, y, h_target / 2]))
    return np.array(points)


target_key_points = generate_target_key_points()
total_required_points = len(target_key_points)  # 37个关键点，全遮才有效


# ==============================================================================
# 3. 核心工具函数（适配A题(3).pdf遮蔽判定与物理运动规则）
# ==============================================================================
def get_missile_position(t):
    """计算t时刻导弹M1的位置（A题(3).pdf匀速直线运动）"""
    return pos_m1_init + vel_m1_vector * t


def get_cloud_center(t, t_detonate, pos_detonate):
    """计算t时刻烟幕云团中心位置（A题(3).pdf匀速下沉规则）"""
    if t < t_detonate:
        return None  # 烟幕未起爆，无有效云团
    return pos_detonate - np.array([0, 0, v_cloud_sink * (t - t_detonate)])


def is_point_covered(missile_pos, target_point, cloud_pos):
    """判断单个目标点是否被烟幕遮蔽（A题(3).pdf 10m有效范围）"""
    if cloud_pos is None:
        return False
    # 计算导弹-目标点视线与烟幕的空间关系
    line_vec = missile_pos - target_point
    cloud_vec = cloud_pos - target_point
    proj = np.dot(cloud_vec, line_vec) / (np.dot(line_vec, line_vec) + 1e-12)  # 避免分母为0
    proj_clamped = np.clip(proj, 0, 1)  # 投影到视线线段上（不超出导弹-目标范围）
    closest_point = target_point + proj_clamped * line_vec  # 视线与烟幕的最近点
    return np.linalg.norm(closest_point - cloud_pos) <= r_cloud


def evaluate_cloud_effect(t_detonate, pos_detonate):
    """评估单枚烟幕的有效遮蔽区间（A题(3).pdf：全遮+20s内+导弹命中前）"""
    t_start = max(t_detonate, 0)
    t_end = min(t_detonate + t_cloud_max, t_m1_fly)  # 有效时间不超过导弹命中假目标
    if t_end - t_start < 0.1:
        return []  # 有效窗口过短，视为无效烟幕

    effective_intervals = []  # 存储有效遮蔽时间段（start, end）
    current_t = t_start
    while current_t <= t_end:
        missile_pos = get_missile_position(current_t)
        cloud_pos = get_cloud_center(current_t, t_detonate, pos_detonate)

        if cloud_pos is not None:
            # 检查是否覆盖所有37个关键点（A题(3).pdf：需完全遮蔽才有效）
            all_covered = True
            for point in target_key_points:
                if not is_point_covered(missile_pos, point, cloud_pos):
                    all_covered = False
                    break
            if all_covered:
                effective_intervals.append((current_t, current_t + 0.1))

        current_t += 0.1  # 时间步长0.1s，平衡精度与效率
    return effective_intervals


def calculate_total_effective_time(intervals_list):
    """合并多枚烟幕的有效区间，计算总遮蔽时间（去重重叠区间）"""
    all_intervals = []
    for intervals in intervals_list:
        all_intervals.extend(intervals)
    if not all_intervals:
        return 0.0  # 无有效遮蔽时间

    # 按起始时间排序，合并重叠或相邻区间
    all_intervals.sort()
    merged = [all_intervals[0]]
    for current in all_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1e-6:  # 重叠或相邻（误差容忍）
            merged[-1] = (last[0], max(last[1], current[1]))  # 合并区间
        else:
            merged.append(current)
    return sum(end - start for start, end in merged)


# ==============================================================================
# 4. 目标函数（强化约束惩罚，适配A题(3).pdf多弹投放规则）
# ==============================================================================
def objective_function(strategy):
    """
    策略变量：[无人机速度, 无人机角度, 烟幕1投放时间, 烟幕1延时, 烟幕2投放时间, 烟幕2延时, 烟幕3投放时间, 烟幕3延时]
    返回值：负的总遮蔽时间（PSO最小化→实际最大化遮蔽时间）
    """
    v_fy1, theta_deg = strategy[0], strategy[1]
    t_drop1, t_delay1 = strategy[2], strategy[3]
    t_drop2, t_delay2 = strategy[4], strategy[5]
    t_drop3, t_delay3 = strategy[6], strategy[7]

    # 约束1：3枚烟幕弹投放间隔≥1s（A题(3).pdf硬性要求，违规则大幅惩罚）
    if (t_drop2 - t_drop1 < min_drop_interval - 1e-6 or
            t_drop3 - t_drop2 < min_drop_interval - 1e-6):
        return 1e12  # 惩罚值足够大，PSO会优先排除该策略

    # 约束2：无人机速度在70~140m/s（A题(3).pdf性能范围）
    if not (v_fy1_min <= v_fy1 <= v_fy1_max):
        return 1e12

    # 约束3：投放时间/引信延时为正（物理意义，避免无效参数）
    if any(t <= 0 for t in [t_drop1, t_delay1, t_drop2, t_delay2, t_drop3, t_delay3]):
        return 1e12

    # 计算无人机速度向量（A题(3).pdf：等高度飞行，z分量为0）
    theta_rad = np.radians(theta_deg)
    vel_fy1 = np.array([v_fy1 * np.cos(theta_rad), v_fy1 * np.sin(theta_rad), 0])

    # 计算单枚烟幕的起爆参数与有效区间
    def process_cloud(t_drop, t_delay):
        # 投放点：无人机飞行t_drop时间后的位置（A题(3).pdf匀速直线飞行）
        pos_drop = pos_fy1_init + vel_fy1 * t_drop
        # 烟幕弹平抛运动（A题(3).pdf：脱离无人机后仅受重力）
        dx = vel_fy1[0] * t_delay  # x方向：随无人机速度
        dy = vel_fy1[1] * t_delay  # y方向：随无人机速度
        dz = -0.5 * G * t_delay ** 2  # z方向：重力下落（向下为负）
        pos_detonate = pos_drop + np.array([dx, dy, dz])  # 烟幕起爆点坐标
        t_detonate = t_drop + t_delay  # 烟幕起爆时间

        # 约束4：起爆时间早于导弹命中假目标（A题(3).pdf：起爆晚则无效）
        if t_detonate >= t_m1_fly - 1e-6:
            return []
        # 评估该枚烟幕的有效遮蔽区间
        return evaluate_cloud_effect(t_detonate, pos_detonate)

    # 处理3枚烟幕弹，获取各自有效区间
    intervals1 = process_cloud(t_drop1, t_delay1)
    intervals2 = process_cloud(t_drop2, t_delay2)
    intervals3 = process_cloud(t_drop3, t_delay3)

    # 计算3枚烟幕的总有效遮蔽时间
    total_time = calculate_total_effective_time([intervals1, intervals2, intervals3])
    return -total_time  # PSO为最小化算法，返回负值实现“最大化总时间”


# ==============================================================================
# 5. PSO优化三大核心增强（拉丁超立方+动态权重+智能扰动）
# ==============================================================================
def enhanced_latin_hypercube_sampling(n_particles, dim, bounds):
    """
    增强版拉丁超立方抽样（适配A题(3).pdf 8维决策变量）：
    1. 每个维度分n_particles段，每段仅1个点，确保均匀覆盖参数范围
    2. 随机打乱区间顺序，避免维度间参数耦合导致的搜索盲区
    """
    particles_pos = np.zeros((n_particles, dim))
    for i in range(dim):
        min_b, max_b = bounds[i]
        # 按粒子数分段，生成区间边界
        segments = np.linspace(min_b, max_b, n_particles + 1)
        # 随机打乱区间索引，增强全局覆盖性
        interval_indices = np.random.permutation(n_particles)
        # 每个区间内随机取点，确保粒子均匀分布
        for j in range(n_particles):
            interval_idx = interval_indices[j]
            particles_pos[j, i] = np.random.uniform(segments[interval_idx], segments[interval_idx + 1])
    return particles_pos


def adaptive_inertia_weight(iter, max_iter, w_max=0.95, w_min=0.35, gbest_history=None):
    """
    自适应动态惯性权重（适配A题(3).pdf多弹优化的复杂空间）：
    1. 基础线性递减：前期大权重（全局探索），后期小权重（局部收敛）
    2. 停滞反馈调整：若全局最优停滞，临时提升权重以跳出局部最优
    """
    # 基础线性递减权重
    base_w = w_max - (w_max - w_min) * (iter / max_iter)
    # 全局最优停滞判断（基于历史记录）
    if gbest_history is not None and len(gbest_history) >= 5:
        recent_gbests = gbest_history[-5:]
        # 若最近5次迭代全局最优无显著提升（阈值1e-8），临时增加权重
        if max(recent_gbests) - min(recent_gbests) < 1e-8:
            base_w = min(w_max, base_w + 0.1)  # 权重上限不超过w_max
    return base_w


def intelligent_position_perturbation(particles_pos, particles_vel, pbest_fitness, iter, max_iter, bounds):
    """
    智能位置扰动（适配A题(3).pdf多弹优化的局部最优问题）：
    1. 分阶段扰动：70%迭代前不扰，70%-90%低概率，90%后高概率
    2. 针对性扰动：仅对未改进个体最优的粒子扰动，节省计算资源
    3. 动态幅度：边界范围的5%-10%，平衡探索精度与稳定性
    """
    n_particles, dim = particles_pos.shape
    # 分阶段设置扰动概率
    if iter < 0.7 * max_iter:
        perturb_prob = 0.0  # 前期稳定探索，不扰动
    elif 0.7 * max_iter <= iter < 0.9 * max_iter:
        perturb_prob = 0.08  # 中期低概率扰动，避免过早打乱
    else:
        perturb_prob = 0.15  # 后期高概率扰动，强化全局探索

    for j in range(n_particles):
        # 计算当前粒子的适应度，判断是否需要扰动
        current_fitness = objective_function(particles_pos[j])
        # 仅对“未改进个体最优”的粒子进行扰动（误差容忍1e-8）
        if current_fitness >= pbest_fitness[j] + 1e-8:
            if np.random.rand() < perturb_prob:
                for k in range(dim):
                    min_b, max_b = bounds[k]
                    # 动态扰动幅度：边界范围的5%-10%
                    perturb_scale = np.random.uniform(0.05, 0.1)
                    perturb = np.random.uniform(-perturb_scale * (max_b - min_b), perturb_scale * (max_b - min_b))
                    particles_pos[j, k] += perturb
                    # 扰动后检查边界，确保符合A题(3).pdf物理约束
                    if particles_pos[j, k] < min_b:
                        particles_pos[j, k] = 2 * min_b - particles_pos[j, k]  # 反弹回边界
                        particles_vel[j, k] *= -0.6  # 速度反向衰减，避免反复越界
                    elif particles_pos[j, k] > max_b:
                        particles_pos[j, k] = 2 * max_b - particles_pos[j, k]
                        particles_vel[j, k] *= -0.6
    return particles_pos, particles_vel

# ==============================================================================
# 6. PSO主程序（整合优化，输出A题(3).pdf问题3最优策略）
# ==============================================================================
# ==============================================================================
# 6. PSO主程序（仅主函数，适配A题(3).pdf问题3：FY1投放3枚烟幕弹干扰M1）
# ==============================================================================
if __name__ == "__main__":
    # PSO参数配置（严格对齐A题(3).pdf约束与多弹优化需求）
    n_particles = 220  # 粒子数：适配8维变量，确保搜索全面性
    max_iter = 350     # 迭代次数：保障复杂策略收敛
    dim = 8            # 决策变量维度：无人机2个（速度/角度）+3枚弹各2个（投放时间/延时）
    w_max, w_min = 0.95, 0.35  # 惯性权重范围：增强全局探索与局部收敛平衡
    c1, c2 = 1.6, 1.8          # 学习因子：强化社会学习，引导向全局最优靠拢
    # 决策变量边界（A题(3).pdf物理约束：速度70-140m/s、投放间隔≥1s等）
    bounds = [
        (70, 140),          # 无人机飞行速度 (m/s)
        (0, 360),           # 无人机飞行角度 (度)
        (0.1, 45.0),        # 第1枚烟幕弹投放时间 (s)
        (0.1, 18.0),        # 第1枚烟幕弹引信延时 (s)
        (1.1, 46.0),        # 第2枚烟幕弹投放时间 (s)：≥第1枚+1s
        (0.1, 18.0),        # 第2枚烟幕弹引信延时 (s)
        (2.1, 47.0),        # 第3枚烟幕弹投放时间 (s)：≥第2枚+1s
        (0.1, 18.0)         # 第3枚烟幕弹引信延时 (s)
    ]

    # 步骤1：初始化粒子群（增强版拉丁超立方抽样，A题(3).pdf多维度适配）
    particles_pos = enhanced_latin_hypercube_sampling(n_particles, dim, bounds)
    # 初始化粒子速度（边界范围的±10%，确保初始速度合理）
    particles_vel = np.zeros((n_particles, dim))
    for i in range(dim):
        min_b, max_b = bounds[i]
        vel_range = 0.1 * (max_b - min_b)
        particles_vel[:, i] = np.random.uniform(-vel_range, vel_range, n_particles)

    # 步骤2：初始化个体最优（pbest）与全局最优（gbest）
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([objective_function(pos) for pos in tqdm(particles_pos, desc="初始化个体最优")])
    gbest_idx = np.argmin(pbest_fitness)
    gbest_pos = np.copy(pbest_pos[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]
    gbest_history = [gbest_fitness]  # 记录全局最优历史，用于自适应权重

    # 步骤3：迭代寻优（整合动态权重+智能扰动，A题(3).pdf遮蔽时间最大化）
    for iter in tqdm(range(max_iter), desc="PSO优化（A题(3).pdf问题3）"):
        # 1. 自适应动态惯性权重（根据迭代进度与全局最优停滞情况调整）
        w = adaptive_inertia_weight(iter, max_iter, w_max, w_min, gbest_history)

        # 2. 遍历粒子，更新速度与位置
        for j in range(n_particles):
            # 速度更新（PSO标准公式：认知项+社会项+惯性项）
            r1, r2 = np.random.rand(2)
            cognitive_term = c1 * r1 * (pbest_pos[j] - particles_pos[j])  # 个体经验
            social_term = c2 * r2 * (gbest_pos - particles_pos[j])        # 全局经验
            particles_vel[j] = w * particles_vel[j] + cognitive_term + social_term

            # 位置更新
            particles_pos[j] += particles_vel[j]

            # 边界处理（反弹逻辑，避免参数超出A题(3).pdf物理约束）
            for k in range(dim):
                min_b, max_b = bounds[k]
                if particles_pos[j, k] < min_b:
                    particles_pos[j, k] = 2 * min_b - particles_pos[j, k]
                    particles_vel[j, k] *= -0.6  # 速度反向衰减，避免反复越界
                elif particles_pos[j, k] > max_b:
                    particles_pos[j, k] = 2 * max_b - particles_pos[j, k]
                    particles_vel[j, k] *= -0.6

        # 3. 智能位置扰动（仅对未改进粒子，A题(3).pdf多弹优化防局部最优）
        particles_pos, particles_vel = intelligent_position_perturbation(
            particles_pos, particles_vel, pbest_fitness, iter, max_iter, bounds
        )

        # 4. 更新个体最优（pbest）
        for j in range(n_particles):
            current_fitness = objective_function(particles_pos[j])
            if current_fitness < pbest_fitness[j] - 1e-8:  # 显著改进才更新
                pbest_fitness[j] = current_fitness
                pbest_pos[j] = np.copy(particles_pos[j])

        # 5. 更新全局最优（gbest）与历史记录
        current_gbest_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_gbest_idx] < gbest_fitness - 1e-8:
            gbest_fitness = pbest_fitness[current_gbest_idx]
            gbest_pos = np.copy(pbest_pos[current_gbest_idx])
        gbest_history.append(gbest_fitness)

        # 6. 中间结果输出（每50次迭代打印，A题(3).pdf优化进度跟踪）
        if (iter + 1) % 50 == 0 or iter == 0:
            current_total_time = -objective_function(gbest_pos)  # 转换为正的遮蔽时间
            print(f"\n迭代{iter + 1:3d}/{max_iter}: A题(3).pdf问题3 当前最优遮蔽时长 = {current_total_time:6.4f}s")

    # 步骤4：最终结果输出（A题(3).pdf问题3最优策略）
    final_total_time = -objective_function(gbest_pos)
    print("\n" + "=" * 70)
    print("A题(3).pdf 问题3 优化完成（FY1投放3枚烟幕弹干扰M1）")
    print("=" * 70)
    print(f"最终最优总有效遮蔽时长: {final_total_time:.4f} 秒")
    print("\n最优投放策略（严格符合A题(3).pdf约束）:")
    print(f"1. 无人机FY1参数:")
    print(f"   - 飞行速度: {gbest_pos[0]:.2f} m/s")
    print(f"   - 飞行角度: {gbest_pos[1]:.2f} 度")
    print(f"\n2. 第1枚烟幕弹:")
    print(f"   - 投放时间: {gbest_pos[2]:.2f} s")
    print(f"   - 引信延时: {gbest_pos[3]:.2f} s")
    print(f"\n3. 第2枚烟幕弹:")
    print(f"   - 投放时间: {gbest_pos[4]:.2f} s")
    print(f"   - 引信延时: {gbest_pos[5]:.2f} s")
    print(f"\n4. 第3枚烟幕弹:")
    print(f"   - 投放时间: {gbest_pos[6]:.2f} s")
    print(f"   - 引信延时: {gbest_pos[7]:.2f} s")
    print("=" * 70)