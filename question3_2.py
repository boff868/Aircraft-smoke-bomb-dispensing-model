import numpy as np
from tqdm import tqdm

# ==============================================================================
# 1. 常量与环境参数
# ==============================================================================
G = 9.8  # 重力加速度(m/s²)

# 目标与武器参数
POS_true_target = np.array([0, 200, 5])  # 真目标中心
r_target = 7.0  # 真目标半径(m)
h_target = 10.0  # 真目标高度(m)
pos_m1_init = np.array([20000, 0, 2000])  # 导弹初始位置
v_m1 = 300.0  # 导弹速度(m/s)
pos_fy1_init = np.array([17800, 0, 1800])  # 无人机初始位置
v_fy1_min, v_fy1_max = 70.0, 140.0  # 无人机速度范围

# 烟幕参数
r_cloud = 10.0  # 烟幕有效半径(m)
v_cloud_sink = 3.0  # 下沉速度(m/s)
t_cloud_max = 20.0  # 单弹最大有效时长(s)
min_drop_interval = 1.0  # 最小投放间隔(s)

# 导弹轨迹预计算
dir_m1 = -pos_m1_init  # 导弹飞行方向向量
vel_m1_vector = v_m1 * dir_m1 / np.linalg.norm(dir_m1)
t_m1_fly = np.linalg.norm(dir_m1) / v_m1  # 导弹总飞行时间(≈67.33s)


# 真目标关键点位采样（覆盖所有关键区域）
def generate_target_key_points():
    """生成真目标的关键采样点（覆盖中心、底面、顶面、侧面边缘）"""
    points = []
    # 1. 目标中心点（1个）
    points.append(POS_true_target)

    # 2. 底面圆周点（12个，每30度1个）
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = r_target * np.cos(theta)
        y = r_target * np.sin(theta) + 200  # 真目标y轴中心为200
        points.append(np.array([x, y, 0]))  # 底面z=0

    # 3. 顶面圆周点（12个）
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = r_target * np.cos(theta)
        y = r_target * np.sin(theta) + 200
        points.append(np.array([x, y, h_target]))  # 顶面z=10

    # 4. 侧面中间点（12个）
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = r_target * np.cos(theta)
        y = r_target * np.sin(theta) + 200
        points.append(np.array([x, y, h_target / 2]))  # 中间z=5

    return np.array(points)


target_key_points = generate_target_key_points()  # 37个关键点位
total_required_points = len(target_key_points)  # 必须覆盖全部点位才有效


# ==============================================================================
# 2. 核心工具函数（全遮盖判定逻辑）
# ==============================================================================
def get_missile_position(t):
    """计算t时刻导弹位置"""
    return pos_m1_init + vel_m1_vector * t


def get_cloud_center(t, t_detonate, pos_detonate):
    """计算t时刻烟幕中心位置（考虑下沉）"""
    if t < t_detonate:
        return None
    return pos_detonate - np.array([0, 0, v_cloud_sink * (t - t_detonate)])


def is_point_covered(missile_pos, target_point, cloud_pos):
    """判断单个目标点是否被烟幕遮盖"""
    if cloud_pos is None:
        return False
    # 计算视线与烟幕的空间关系
    line_vec = missile_pos - target_point
    cloud_vec = cloud_pos - target_point
    proj = np.dot(cloud_vec, line_vec) / (np.dot(line_vec, line_vec) + 1e-12)
    proj_clamped = np.clip(proj, 0, 1)  # 投影到视线线段上
    closest_point = target_point + proj_clamped * line_vec
    return np.linalg.norm(closest_point - cloud_pos) <= r_cloud


def evaluate_cloud_effect(t_detonate, pos_detonate):
    """评估单枚烟幕的效果：仅当覆盖所有点位时才记录有效时间"""
    t_start = max(t_detonate, 0)
    t_end = min(t_detonate + t_cloud_max, t_m1_fly)
    if t_end - t_start < 0.1:
        return []  # 无效烟幕

    effective_intervals = []  # 有效遮蔽时间段

    # 逐时间步评估
    current_t = t_start
    while current_t <= t_end:
        missile_pos = get_missile_position(current_t)
        cloud_pos = get_cloud_center(current_t, t_detonate, pos_detonate)

        if cloud_pos is not None:
            # 检查是否覆盖所有目标点位
            all_covered = True
            for point in target_key_points:
                if not is_point_covered(missile_pos, point, cloud_pos):
                    all_covered = False
                    break

            # 只有全部覆盖才记录有效时间
            if all_covered:
                effective_intervals.append((current_t, current_t + 0.1))

        current_t += 0.1  # 时间步长0.1s

    return effective_intervals


def calculate_total_effective_time(intervals_list):
    """合并计算总有效遮盖时间（合并重叠区间）"""
    # 合并所有有效区间
    all_intervals = []
    for intervals in intervals_list:
        all_intervals.extend(intervals)

    if not all_intervals:
        return 0.0  # 总时间为0

    # 按时间排序并合并重叠区间
    all_intervals.sort()
    merged = [all_intervals[0]]
    for current in all_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1e-6:  # 重叠或相邻
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return sum(end - start for start, end in merged)


# ==============================================================================
# 3. 目标函数（全遮盖判定）
# ==============================================================================
def objective_function(strategy):
    """
    目标函数：仅当烟幕完全覆盖所有目标点位时才计算有效时间
    返回值：负的总有效时间（用于PSO最小化）
    """
    # 解析策略参数
    v_fy1, theta_deg = strategy[0], strategy[1]
    t_drop1, t_delay1 = strategy[2], strategy[3]
    t_drop2, t_delay2 = strategy[4], strategy[5]
    t_drop3, t_delay3 = strategy[6], strategy[7]

    # 约束1：投放间隔
    if (t_drop2 - t_drop1 < min_drop_interval - 1e-6 or
            t_drop3 - t_drop2 < min_drop_interval - 1e-6):
        return 1e6  # 惩罚：间隔不足

    # 约束2：无人机速度范围
    if not (v_fy1_min <= v_fy1 <= v_fy1_max):
        return 1e6  # 惩罚：速度非法

    # 计算无人机速度向量
    theta_rad = np.radians(theta_deg)
    vel_fy1 = np.array([
        v_fy1 * np.cos(theta_rad),
        v_fy1 * np.sin(theta_rad),
        0
    ])

    # 计算单枚烟幕的起爆参数与效果
    def process_cloud(t_drop, t_delay):
        # 投放点坐标
        pos_drop = pos_fy1_init + vel_fy1 * t_drop
        # 起爆点坐标（平抛运动）
        dx = vel_fy1[0] * t_delay
        dy = vel_fy1[1] * t_delay
        dz = -0.5 * G * t_delay ** 2  # 重力下落
        pos_detonate = pos_drop + np.array([dx, dy, dz])
        t_detonate = t_drop + t_delay

        # 评估烟幕效果（全遮盖才有效）
        return evaluate_cloud_effect(t_detonate, pos_detonate)

    # 处理三枚烟幕
    intervals1 = process_cloud(t_drop1, t_delay1)
    intervals2 = process_cloud(t_drop2, t_delay2)
    intervals3 = process_cloud(t_drop3, t_delay3)

    # 计算总有效时间
    total_time = calculate_total_effective_time([intervals1, intervals2, intervals3])

    # PSO最小化负时间 = 最大化实际时间
    return -total_time


# ==============================================================================
# 4. PSO实现与结果输出
# ==============================================================================
def get_total_coverage_time(strategy):
    """计算给定策略的总遮盖时间"""
    v_fy1, theta_deg = strategy[0], strategy[1]
    t_drop1, t_delay1 = strategy[2], strategy[3]
    t_drop2, t_delay2 = strategy[4], strategy[5]
    t_drop3, t_delay3 = strategy[6], strategy[7]

    theta_rad = np.radians(theta_deg)
    vel_fy1 = np.array([
        v_fy1 * np.cos(theta_rad),
        v_fy1 * np.sin(theta_rad),
        0
    ])

    def process_cloud(t_drop, t_delay):
        pos_drop = pos_fy1_init + vel_fy1 * t_drop
        dx = vel_fy1[0] * t_delay
        dy = vel_fy1[1] * t_delay
        dz = -0.5 * G * t_delay ** 2
        pos_detonate = pos_drop + np.array([dx, dy, dz])
        t_detonate = t_drop + t_delay
        return evaluate_cloud_effect(t_detonate, pos_detonate)

    intervals1 = process_cloud(t_drop1, t_delay1)
    intervals2 = process_cloud(t_drop2, t_delay2)
    intervals3 = process_cloud(t_drop3, t_delay3)

    return calculate_total_effective_time([intervals1, intervals2, intervals3])


# PSO优化主程序
if __name__ == "__main__":
    # PSO参数配置
    n_particles = 200  # 增加粒子数量提高搜索能力
    max_iter = 300  # 增加迭代次数确保收敛
    dim = 8  # 决策变量维度
    w_max, w_min = 0.9, 0.4  # 惯性权重范围
    c1, c2 = 1.5, 1.5  # 学习因子

    # 决策变量边界
    bounds = [
        (70, 140),  # 无人机速度
        (0, 360),  # 无人机方向角
        (0.1, 50.0),  # 第1枚投放时间
        (0.1, 20.0),  # 第1枚引信延时
        (1.1, 51.0),  # 第2枚投放时间（≥第1枚+1s）
        (0.1, 20.0),  # 第2枚引信延时
        (2.1, 52.0),  # 第3枚投放时间（≥第2枚+1s）
        (0.1, 20.0)  # 第3枚引信延时
    ]

    # 初始化粒子群
    particles_pos = np.zeros((n_particles, dim))
    for i in range(dim):
        min_b, max_b = bounds[i]
        segments = np.linspace(min_b, max_b, n_particles + 1)
        for j in range(n_particles):
            particles_pos[j, i] = np.random.uniform(segments[j], segments[j + 1])

    # 初始化速度
    particles_vel = np.zeros((n_particles, dim))
    for i in range(dim):
        min_b, max_b = bounds[i]
        particles_vel[:, i] = np.random.uniform(
            -0.1 * (max_b - min_b),
            0.1 * (max_b - min_b),
            n_particles
        )

    # 初始化最优解
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([objective_function(pos) for pos in particles_pos])
    gbest_idx = np.argmin(pbest_fitness)
    gbest_pos = np.copy(pbest_pos[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]

    # 迭代优化
    for iter in tqdm(range(max_iter), desc="优化中"):
        w = w_max - (w_max - w_min) * (iter / max_iter)

        for j in range(n_particles):
            r1, r2 = np.random.rand(2)

            # 更新速度
            cognitive = c1 * r1 * (pbest_pos[j] - particles_pos[j])
            social = c2 * r2 * (gbest_pos - particles_pos[j])
            particles_vel[j] = w * particles_vel[j] + cognitive + social

            # 更新位置并处理边界
            particles_pos[j] += particles_vel[j]
            for k in range(dim):
                min_b, max_b = bounds[k]
                if particles_pos[j, k] < min_b:
                    particles_pos[j, k] = 2 * min_b - particles_pos[j, k]
                    particles_vel[j, k] *= -0.5
                elif particles_pos[j, k] > max_b:
                    particles_pos[j, k] = 2 * max_b - particles_pos[j, k]
                    particles_vel[j, k] *= -0.5

            # 后期增加扰动概率以跳出局部最优
            if iter > 0.7 * max_iter and np.random.rand() < 0.1:
                for k in range(dim):
                    min_b, max_b = bounds[k]
                    particles_pos[j, k] += np.random.uniform(
                        -0.08 * (max_b - min_b),
                        0.08 * (max_b - min_b)
                    )

        # 更新最优解
        for j in range(n_particles):
            current_fitness = objective_function(particles_pos[j])
            if current_fitness < pbest_fitness[j] - 1e-6:
                pbest_fitness[j] = current_fitness
                pbest_pos[j] = np.copy(particles_pos[j])

        current_gbest_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_gbest_idx] < gbest_fitness - 1e-6:
            gbest_fitness = pbest_fitness[current_gbest_idx]
            gbest_pos = np.copy(pbest_pos[current_gbest_idx])

        # 输出中间结果
        if (iter + 1) % 50 == 0:
            current_total = get_total_coverage_time(gbest_pos)
            print(f"\n迭代{iter + 1}/{max_iter}: 当前总遮盖时间={current_total:.2f}s")

    # 计算并输出最终总遮盖时间
    final_total_time = get_total_coverage_time(gbest_pos)
    print("\n" + "=" * 50)
    print(f"最终优化得到的总遮盖时间为: {final_total_time:.4f} 秒")
    print("=" * 50)
