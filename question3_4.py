import numpy as np
from tqdm import tqdm

# ==============================================================================
# 1. 常量与环境参数（严格对齐文档）
# ==============================================================================
G = 9.8  # 重力加速度(m/s²)
POS_true_target = np.array([0, 200, 5])  # 真目标中心
r_target = 7.0  # 真目标半径(m)
h_target = 10.0  # 真目标高度(m)
pos_m1_init = np.array([20000, 0, 2000])  # 导弹初始位置
v_m1 = 300.0  # 导弹速度(m/s)
pos_fy1_init = np.array([17800, 0, 1800])  # 无人机初始位置
v_fy1_min, v_fy1_max = 70.0, 140.0  # 无人机速度范围
r_cloud = 10.0  # 烟幕有效半径(m)
v_cloud_sink = 3.0  # 下沉速度(m/s)
t_cloud_max = 20.0  # 单弹最大有效时长(s)
min_drop_interval = 1.0  # 最小投放间隔(s)

# 导弹轨迹预计算
dir_m1 = -pos_m1_init
vel_m1_vector = v_m1 * dir_m1 / np.linalg.norm(dir_m1)
t_m1_fly = np.linalg.norm(dir_m1) / v_m1


# ==============================================================================
# 2. 真目标关键点位采样
# ==============================================================================
def generate_target_key_points():
    points = []
    points.append(POS_true_target)
    # 底面圆周点（12个）
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = r_target * np.cos(theta)
        y = r_target * np.sin(theta) + 200
        points.append(np.array([x, y, 0]))
    # 顶面圆周点（12个）
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = r_target * np.cos(theta)
        y = r_target * np.sin(theta) + 200
        points.append(np.array([x, y, h_target]))
    # 侧面中间点（12个）
    for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x = r_target * np.cos(theta)
        y = r_target * np.sin(theta) + 200
        points.append(np.array([x, y, h_target / 2]))
    return np.array(points)


target_key_points = generate_target_key_points()


# ==============================================================================
# 3. 核心工具函数
# ==============================================================================
def get_missile_position(t):
    return pos_m1_init + vel_m1_vector * t


def get_cloud_center(t, t_detonate, pos_detonate):
    if t < t_detonate:
        return None
    return pos_detonate - np.array([0, 0, v_cloud_sink * (t - t_detonate)])


def is_point_covered(missile_pos, target_point, cloud_pos):
    if cloud_pos is None:
        return False
    line_vec = missile_pos - target_point
    cloud_vec = cloud_pos - target_point
    proj = np.dot(cloud_vec, line_vec) / (np.dot(line_vec, line_vec) + 1e-12)
    proj_clamped = np.clip(proj, 0, 1)
    closest_point = target_point + proj_clamped * line_vec
    return np.linalg.norm(closest_point - cloud_pos) <= r_cloud


def evaluate_cloud_effect(t_detonate, pos_detonate):
    t_start = max(t_detonate, 0)
    t_end = min(t_detonate + t_cloud_max, t_m1_fly)
    if t_end - t_start < 0.1:
        return []
    effective_intervals = []
    current_t = t_start
    while current_t <= t_end:
        missile_pos = get_missile_position(current_t)
        cloud_pos = get_cloud_center(current_t, t_detonate, pos_detonate)
        if cloud_pos is not None:
            all_covered = True
            for point in target_key_points:
                if not is_point_covered(missile_pos, point, cloud_pos):
                    all_covered = False
                    break
            if all_covered:
                effective_intervals.append((current_t, current_t + 0.1))
        current_t += 0.1
    return effective_intervals


def calculate_total_effective_time(intervals_list):
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
# 4. 目标函数
# ==============================================================================
def objective_function(strategy):
    v_fy1, theta_deg = strategy[0], strategy[1]
    t_drop1, t_delay1 = strategy[2], strategy[3]
    t_drop2, t_delay2 = strategy[4], strategy[5]
    t_drop3, t_delay3 = strategy[6], strategy[7]

    if (t_drop2 - t_drop1 < min_drop_interval - 1e-6 or
            t_drop3 - t_drop2 < min_drop_interval - 1e-6):
        return 1e6
    if not (v_fy1_min <= v_fy1 <= v_fy1_max):
        return 1e6

    theta_rad = np.radians(theta_deg)
    vel_fy1 = np.array([v_fy1 * np.cos(theta_rad), v_fy1 * np.sin(theta_rad), 0])

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
    total_time = calculate_total_effective_time([intervals1, intervals2, intervals3])
    return -total_time


# ==============================================================================
# 5. 计算总遮盖时间函数
# ==============================================================================
def get_total_coverage_time(strategy):
    v_fy1, theta_deg = strategy[0], strategy[1]
    t_drop1, t_delay1 = strategy[2], strategy[3]
    t_drop2, t_delay2 = strategy[4], strategy[5]
    t_drop3, t_delay3 = strategy[6], strategy[7]

    theta_rad = np.radians(theta_deg)
    vel_fy1 = np.array([v_fy1 * np.cos(theta_rad), v_fy1 * np.sin(theta_rad), 0])

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


# ==============================================================================
# 6. PSO主程序（打印完整策略信息）
# ==============================================================================
if __name__ == "__main__":
    # PSO参数配置
    n_particles = 200
    max_iter = 300
    dim = 8
    w_max, w_min = 0.9, 0.4
    c1, c2 = 1.5, 1.5
    bounds = [
        (70, 140), (0, 360), (0.1, 50.0), (0.1, 20.0),
        (1.1, 51.0), (0.1, 20.0), (2.1, 52.0), (0.1, 20.0)
    ]

    # 初始化粒子群
    particles_pos = np.zeros((n_particles, dim))
    for i in range(dim):
        min_b, max_b = bounds[i]
        segments = np.linspace(min_b, max_b, n_particles + 1)
        for j in range(n_particles):
            particles_pos[j, i] = np.random.uniform(segments[j], segments[j + 1])

    particles_vel = np.zeros((n_particles, dim))
    for i in range(dim):
        min_b, max_b = bounds[i]
        particles_vel[:, i] = np.random.uniform(-0.1 * (max_b - min_b), 0.1 * (max_b - min_b), n_particles)

    # 初始化最优解
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([objective_function(pos) for pos in particles_pos])
    gbest_idx = np.argmin(pbest_fitness)
    gbest_pos = np.copy(pbest_pos[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]

    # 迭代优化（删除中间输出）
    for iter in tqdm(range(max_iter), desc="优化中"):
        w = w_max - (w_max - w_min) * (iter / max_iter)
        for j in range(n_particles):
            r1, r2 = np.random.rand(2)
            cognitive = c1 * r1 * (pbest_pos[j] - particles_pos[j])
            social = c2 * r2 * (gbest_pos - particles_pos[j])
            particles_vel[j] = w * particles_vel[j] + cognitive + social

            particles_pos[j] += particles_vel[j]
            for k in range(dim):
                min_b, max_b = bounds[k]
                if particles_pos[j, k] < min_b:
                    particles_pos[j, k] = 2 * min_b - particles_pos[j, k]
                    particles_vel[j, k] *= -0.5
                elif particles_pos[j, k] > max_b:
                    particles_pos[j, k] = 2 * max_b - particles_pos[j, k]
                    particles_vel[j, k] *= -0.5

            # 后期扰动
            if iter > 0.7 * max_iter and np.random.rand() < 0.1:
                for k in range(dim):
                    min_b, max_b = bounds[k]
                    particles_pos[j, k] += np.random.uniform(-0.08 * (max_b - min_b), 0.08 * (max_b - min_b))

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

    # 解析并打印完整策略信息
    final_total_time = get_total_coverage_time(gbest_pos)
    print(f"\n最优总有效遮蔽时间: {final_total_time:.4f} 秒")

    # 提取最优策略参数
    v_fy1_opt = gbest_pos[0]
    theta_deg_opt = gbest_pos[1]
    t_drop1_opt = gbest_pos[2]
    t_delay1_opt = gbest_pos[3]
    t_detonate1_opt = t_drop1_opt + t_delay1_opt
    t_drop2_opt = gbest_pos[4]
    t_delay2_opt = gbest_pos[5]
    t_detonate2_opt = t_drop2_opt + t_delay2_opt
    t_drop3_opt = gbest_pos[6]
    t_delay3_opt = gbest_pos[7]
    t_detonate3_opt = t_drop3_opt + t_delay3_opt

    # 计算无人机飞行方向向量
    theta_rad_opt = np.radians(theta_deg_opt)
    vel_fy1_vector_opt = np.array([
        v_fy1_opt * np.cos(theta_rad_opt),
        v_fy1_opt * np.sin(theta_rad_opt),
        0
    ])

    # 计算每枚弹的投放点和起爆点
    theta_rad_opt = np.radians(theta_deg_opt)
    vel_fy1 = np.array([v_fy1_opt * np.cos(theta_rad_opt), v_fy1_opt * np.sin(theta_rad_opt), 0])

    # 第1枚弹
    pos_drop1_opt = pos_fy1_init + vel_fy1 * t_drop1_opt
    dx1 = vel_fy1[0] * t_delay1_opt
    dy1 = vel_fy1[1] * t_delay1_opt
    dz1 = -0.5 * G * t_delay1_opt ** 2
    pos_detonate1_opt = pos_drop1_opt + np.array([dx1, dy1, dz1])

    # 第2枚弹
    pos_drop2_opt = pos_fy1_init + vel_fy1 * t_drop2_opt
    dx2 = vel_fy1[0] * t_delay2_opt
    dy2 = vel_fy1[1] * t_delay2_opt
    dz2 = -0.5 * G * t_delay2_opt ** 2
    pos_detonate2_opt = pos_drop2_opt + np.array([dx2, dy2, dz2])

    # 第3枚弹
    pos_drop3_opt = pos_fy1_init + vel_fy1 * t_drop3_opt
    dx3 = vel_fy1[0] * t_delay3_opt
    dy3 = vel_fy1[1] * t_delay3_opt
    dz3 = -0.5 * G * t_delay3_opt ** 2
    pos_detonate3_opt = pos_drop3_opt + np.array([dx3, dy3, dz3])


    print("\n" + "="*60)
    print("无人机FY1投放3枚烟幕弹的最优策略详情")
    print("="*60)
    print(f"1. 无人机飞行参数:")
    print(f"   - 飞行速度: {v_fy1_opt:.2f} m/s（符合70~140 m/s约束）")
    print(f"   - 飞行方向: 与x轴正方向夹角 {theta_deg_opt:.2f}°（方向向量: [{vel_fy1_vector_opt[0]:.2f}, {vel_fy1_vector_opt[1]:.2f}, 0.00]）")
    print(f"\n2. 第1枚烟幕弹:")
    print(f"   - 投放时间: {t_drop1_opt:.2f} s")
    print(f"   - 起爆时间: {t_detonate1_opt:.2f} s")
    print(f"   - 投放点坐标: [{pos_drop1_opt[0]:.2f}, {pos_drop1_opt[1]:.2f}, {pos_drop1_opt[2]:.2f}]")
    print(f"   - 起爆点坐标: [{pos_detonate1_opt[0]:.2f}, {pos_detonate1_opt[1]:.2f}, {pos_detonate1_opt[2]:.2f}]")
    print(f"\n3. 第2枚烟幕弹:")
    print(f"   - 投放时间: {t_drop2_opt:.2f} s（与第1枚间隔 {t_drop2_opt - t_drop1_opt:.2f} s，符合≥1s约束）")
    print(f"   - 起爆时间: {t_detonate2_opt:.2f} s")
    print(f"   - 投放点坐标: [{pos_drop2_opt[0]:.2f}, {pos_drop2_opt[1]:.2f}, {pos_drop2_opt[2]:.2f}]")
    print(f"   - 起爆点坐标: [{pos_detonate2_opt[0]:.2f}, {pos_detonate2_opt[1]:.2f}, {pos_detonate2_opt[2]:.2f}]")
    print(f"\n4. 第3枚烟幕弹:")
    print(f"   - 投放时间: {t_drop3_opt:.2f} s（与第2枚间隔 {t_drop3_opt - t_drop2_opt:.2f} s，符合≥1s约束）")
    print(f"   - 起爆时间: {t_detonate3_opt:.2f} s")
    print(f"   - 投放点坐标: [{pos_drop3_opt[0]:.2f}, {pos_drop3_opt[1]:.2f}, {pos_drop3_opt[2]:.2f}]")
    print(f"   - 起爆点坐标: [{pos_detonate3_opt[0]:.2f}, {pos_detonate3_opt[1]:.2f}, {pos_detonate3_opt[2]:.2f}]")
    print("=" * 60)