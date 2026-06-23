import numpy as np
from tqdm import tqdm

# ==============================================================================
# 1. 常量与环境设置
# ==============================================================================
G = 9.8  # 重力加速度(m/s²)

# 目标参数
POS_TRUE_TARGET = np.array([0, 200, 5])  # 真目标中心
R_TARGET = 7.0  # 真目标半径(m)
H_TARGET = 10.0  # 真目标高度(m)

# 导弹参数
POS_M1_INIT = np.array([20000, 0, 2000])
V_M1 = 300.0
DIR_M1 = -POS_M1_INIT
VEL_M1_VECTOR = V_M1 * DIR_M1 / np.linalg.norm(DIR_M1)
T_M1_FLY = np.linalg.norm(DIR_M1) / V_M1  # ≈67.33s

# 无人机参数
DRONES = {
    "FY1": {"pos_init": np.array([17800, 0, 1800])},
    "FY2": {"pos_init": np.array([12000, 1400, 1400])},
    "FY3": {"pos_init": np.array([6000, -3000, 700])}
}
V_FY_MIN, V_FY_MAX = 70.0, 140.0

# 烟幕参数
R_CLOUD = 10.0  # 有效半径
V_CLOUD_SINK = 3.0  # 下沉速度
T_CLOUD_MAX = 20.0  # 最大有效时长


# 真目标37个关键点位
def generate_target_key_points():
    points = [POS_TRUE_TARGET]
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = R_TARGET * np.cos(theta)
        y = R_TARGET * np.sin(theta) + 200
        points.append(np.array([x, y, 0]))
        points.append(np.array([x, y, H_TARGET]))
        points.append(np.array([x, y, H_TARGET / 2]))
    return np.array(points)


TARGET_KEY_POINTS = generate_target_key_points()


# ==============================================================================
# 2. 核心工具函数
# ==============================================================================
def get_missile_position(t):
    return POS_M1_INIT + VEL_M1_VECTOR * t


def get_cloud_center(t, t_detonate, pos_detonate):
    if t < t_detonate:
        return None
    # 烟幕随时间下沉
    cloud_z = pos_detonate[2] - V_CLOUD_SINK * (t - t_detonate)
    return np.array([pos_detonate[0], pos_detonate[1], cloud_z])


def is_point_covered(missile_pos, target_point, cloud_pos):
    if cloud_pos is None:
        return False
    line_vec = missile_pos - target_point
    cloud_vec = cloud_pos - target_point
    proj = np.dot(cloud_vec, line_vec) / (np.dot(line_vec, line_vec) + 1e-12)
    proj_clamped = np.clip(proj, 0, 1)
    closest_point = target_point + proj_clamped * line_vec
    return np.linalg.norm(closest_point - cloud_pos) <= R_CLOUD


def evaluate_cloud_effect(t_detonate, pos_detonate):
    t_start = max(t_detonate, 0)
    t_end = min(t_detonate + T_CLOUD_MAX, T_M1_FLY)
    if t_end - t_start < 0.1:
        return []

    effective_intervals = []
    current_t = t_start
    while current_t <= t_end:
        missile_pos = get_missile_position(current_t)
        cloud_pos = get_cloud_center(current_t, t_detonate, pos_detonate)

        if cloud_pos is not None:
            all_covered = True
            for point in TARGET_KEY_POINTS:
                if not is_point_covered(missile_pos, point, cloud_pos):
                    all_covered = False
                    break
            if all_covered:
                effective_intervals.append((current_t, current_t + 0.1))

        current_t += 0.1
    return effective_intervals


def calculate_total_effective_time(intervals_list):
    """合并时间区间，计算去重后的总时长"""
    all_intervals = []
    for intervals in intervals_list:
        all_intervals.extend(intervals)
    if not all_intervals:
        return 0.0

    all_intervals.sort()
    merged = [all_intervals[0]]
    for current in all_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1e-6:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return sum(end - start for start, end in merged)


# ==============================================================================
# 3. 目标函数（仅以总遮盖时间为优化目标）
# ==============================================================================
def objective_function(strategy):
    # 解析12维策略参数
    fy1_v, fy1_theta = strategy[0], strategy[1]
    fy1_t_drop, fy1_t_delay = strategy[2], strategy[3]
    fy2_v, fy2_theta = strategy[4], strategy[5]
    fy2_t_drop, fy2_t_delay = strategy[6], strategy[7]
    fy3_v, fy3_theta = strategy[8], strategy[9]
    fy3_t_drop, fy3_t_delay = strategy[10], strategy[11]

    # 基础约束：速度范围
    if not (V_FY_MIN <= fy1_v <= V_FY_MAX and
            V_FY_MIN <= fy2_v <= V_FY_MAX and
            V_FY_MIN <= fy3_v <= V_FY_MAX):
        return 1e6

    # 计算各烟幕起爆时间和位置
    def get_cloud_params(drone_name, v, theta_deg, t_drop, t_delay):
        theta_rad = np.radians(theta_deg)
        vel = np.array([v * np.cos(theta_rad), v * np.sin(theta_rad), 0])
        pos_drop = DRONES[drone_name]["pos_init"] + vel * t_drop
        dx = vel[0] * t_delay
        dy = vel[1] * t_delay
        dz = -0.5 * G * t_delay ** 2
        pos_detonate = pos_drop + np.array([dx, dy, dz])
        t_detonate = t_drop + t_delay
        return t_detonate, pos_detonate

    # 获取三枚烟幕的核心参数
    t1, pos1 = get_cloud_params("FY1", fy1_v, fy1_theta, fy1_t_drop, fy1_t_delay)
    t2, pos2 = get_cloud_params("FY2", fy2_v, fy2_theta, fy2_t_drop, fy2_t_delay)
    t3, pos3 = get_cloud_params("FY3", fy3_v, fy3_theta, fy3_t_drop, fy3_t_delay)

    # 基础约束：起爆时间需早于导弹命中
    if t1 >= T_M1_FLY - 1e-6 or t2 >= T_M1_FLY - 1e-6 or t3 >= T_M1_FLY - 1e-6:
        return 1e6

    # 评估各烟幕有效时段（无任何奖励机制）
    intervals1 = evaluate_cloud_effect(t1, pos1)
    intervals2 = evaluate_cloud_effect(t2, pos2)
    intervals3 = evaluate_cloud_effect(t3, pos3)

    # 总有效时间（唯一优化目标）
    total_time = calculate_total_effective_time([intervals1, intervals2, intervals3])

    # PSO最小化目标：直接返回负的总时间（最大化总时间等价于最小化负总时间）
    return -total_time


# ==============================================================================
# 4. 结果计算函数
# ==============================================================================
def get_final_total_time(strategy):
    fy1_v, fy1_theta = strategy[0], strategy[1]
    fy1_t_drop, fy1_t_delay = strategy[2], strategy[3]
    fy2_v, fy2_theta = strategy[4], strategy[5]
    fy2_t_drop, fy2_t_delay = strategy[6], strategy[7]
    fy3_v, fy3_theta = strategy[8], strategy[9]
    fy3_t_drop, fy3_t_delay = strategy[10], strategy[11]

    def get_intervals(drone_name, v, theta_deg, t_drop, t_delay):
        theta_rad = np.radians(theta_deg)
        vel = np.array([v * np.cos(theta_rad), v * np.sin(theta_rad), 0])
        pos_drop = DRONES[drone_name]["pos_init"] + vel * t_drop
        dx = vel[0] * t_delay
        dy = vel[1] * t_delay
        dz = -0.5 * G * t_delay ** 2
        pos_detonate = pos_drop + np.array([dx, dy, dz])
        t_detonate = t_drop + t_delay
        return evaluate_cloud_effect(t_detonate, pos_detonate)

    intervals1 = get_intervals("FY1", fy1_v, fy1_theta, fy1_t_drop, fy1_t_delay)
    intervals2 = get_intervals("FY2", fy2_v, fy2_theta, fy2_t_drop, fy2_t_delay)
    intervals3 = get_intervals("FY3", fy3_v, fy3_theta, fy3_t_drop, fy3_t_delay)

    return calculate_total_effective_time([intervals1, intervals2, intervals3])


# ==============================================================================
# 5. PSO算法实现（增强全局搜索能力）
# ==============================================================================
if __name__ == "__main__":
    # PSO参数（强化全局搜索，避免陷入局部最优）
    N_PARTICLES = 400  # 更多粒子探索不同策略
    MAX_ITER = 50     # 更多迭代次数确保收敛
    DIM = 12
    W_MAX, W_MIN = 0.95, 0.3  # 更大的惯性权重范围，增强探索
    C1, C2 = 1.5, 1.5  # 平衡个体与群体学习
    PERTURB_PROB = 0.25  # 更高的扰动概率
    PERTURB_SCALE = 0.15  # 更大的扰动幅度

    # 变量边界（放宽部分限制，扩大搜索空间）
    BOUNDS = [
        (70, 140), (0, 360), (0.1, 30.0), (0.1, 20.0),  # FY1
        (70, 140), (0, 360), (0.1, 40.0), (0.1, 20.0),  # FY2
        (70, 140), (0, 360), (0.1, 50.0), (0.1, 20.0)   # FY3
    ]

    # 初始化粒子群（拉丁超立方抽样）
    particles_pos = np.zeros((N_PARTICLES, DIM))
    for i in range(DIM):
        min_b, max_b = BOUNDS[i]
        segments = np.linspace(min_b, max_b, N_PARTICLES + 1)
        for j in range(N_PARTICLES):
            particles_pos[j, i] = np.random.uniform(segments[j], segments[j + 1])

    particles_vel = np.zeros((N_PARTICLES, DIM))
    for i in range(DIM):
        min_b, max_b = BOUNDS[i]
        particles_vel[:, i] = np.random.uniform(-0.15 * (max_b - min_b), 0.15 * (max_b - min_b), N_PARTICLES)

    # 初始化最优解
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([objective_function(pos) for pos in particles_pos])
    gbest_idx = np.argmin(pbest_fitness)
    gbest_pos = np.copy(pbest_pos[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]

    # 迭代寻优
    for iter in tqdm(range(MAX_ITER), desc="优化中（仅最大化遮盖时间）"):
        W = W_MAX - (W_MAX - W_MIN) * (iter / MAX_ITER)

        for j in range(N_PARTICLES):
            # 速度更新
            r1, r2 = np.random.rand(2)
            cognitive = C1 * r1 * (pbest_pos[j] - particles_pos[j])
            social = C2 * r2 * (gbest_pos - particles_pos[j])
            particles_vel[j] = W * particles_vel[j] + cognitive + social

            # 位置更新与边界处理
            particles_pos[j] += particles_vel[j]
            for k in range(DIM):
                min_b, max_b = BOUNDS[k]
                if particles_pos[j, k] < min_b:
                    particles_pos[j, k] = 2 * min_b - particles_pos[j, k]
                    particles_vel[j, k] *= -0.5
                elif particles_pos[j, k] > max_b:
                    particles_pos[j, k] = 2 * max_b - particles_pos[j, k]
                    particles_vel[j, k] *= -0.5

            # 后期强化扰动（避免陷入局部最优）
            if iter > 0.5 * MAX_ITER and np.random.rand() < PERTURB_PROB:
                for k in range(DIM):
                    min_b, max_b = BOUNDS[k]
                    particles_pos[j, k] += np.random.uniform(-PERTURB_SCALE * (max_b - min_b),
                                                             PERTURB_SCALE * (max_b - min_b))

            # 更新个体最优
            current_fitness = objective_function(particles_pos[j])
            if current_fitness < pbest_fitness[j] - 1e-6:
                pbest_fitness[j] = current_fitness
                pbest_pos[j] = np.copy(particles_pos[j])

        # 更新全局最优
        current_gbest_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_gbest_idx] < gbest_fitness - 1e-6:
            gbest_fitness = pbest_fitness[current_gbest_idx]
            gbest_pos = np.copy(pbest_pos[current_gbest_idx])

    # 输出最终结果
    final_max_time = get_final_total_time(gbest_pos)
    print(f"\n最大遮盖时间为: {final_max_time:.4f} 秒")
