import math

# 一、基础参数定义（严格匹配文档问题五“多机多弹”场景）
# 1. 导弹参数（文档“问题重述”中初始位置、速度等）
missiles = {
    "M1": {"init_pos": (20000, 0, 2000), "speed": 300, "direction": (-1, 0, 0)},
    "M2": {"init_pos": (19000, 600, 2100), "speed": 300, "direction": (-0.98, -0.06, 0)},
    "M3": {"init_pos": (18000, -600, 1900), "speed": 300, "direction": (-0.97, 0.07, 0)}
}

# 2. 无人机参数（文档“问题重述”中初始位置、性能约束）
uavs = {
    "FY1": {"init_pos": (17800, 0, 1800), "max_speed": 140, "max_bombs": 3},
    "FY2": {"init_pos": (12000, 1400, 1400), "max_speed": 140, "max_bombs": 3},
    "FY3": {"init_pos": (6000, -3000, 700), "max_speed": 140, "max_bombs": 3},
    "FY4": {"init_pos": (11000, 2000, 1800), "max_speed": 140, "max_bombs": 3},
    "FY5": {"init_pos": (13000, -2000, 1300), "max_speed": 140, "max_bombs": 3}
}

# 3. 真目标与烟幕参数（文档“模型假设”“符号说明”，补充起爆延迟）
real_target = (0, 200, 5)  # 真目标下底面圆心(0,200,0)，取中心高度5m
smoke_params = {
    "buffer_time": 2,          # 投弹缓冲时间（文档“威胁窗口模型”）
    "single_bomb_duration": 10,# 单枚弹有效时间（文档T_cloud_duration=20s，取核心有效时段）
    "bomb_interval": 1,        # 投弹间隔（文档“多弹投放约束”）
    "detonation_delay": 2.0    # 投弹到起爆延迟（文档问题一“间隔3.6秒起爆”逻辑衍生）
}

# 4. 最优分配方案（文档问题五“权重比例资源分配模型”结果）
optimal_allocation = {
    "M3": [
        {"uav_id": "FY4", "assigned_bombs": 3, "real_time_weight": 0.3821},
        {"uav_id": "FY3", "assigned_bombs": 2, "real_time_weight": 0.3675}
    ],
    "M2": [
        {"uav_id": "FY5", "assigned_bombs": 3, "real_time_weight": 0.3742},
        {"uav_id": "FY2", "assigned_bombs": 2, "real_time_weight": 0.3518},
        {"uav_id": "FY3", "assigned_bombs": 1, "real_time_weight": 0.3286}
    ],
    "M1": [
        {"uav_id": "FY1", "assigned_bombs": 3, "real_time_weight": 0.4120},
        {"uav_id": "FY2", "assigned_bombs": 1, "real_time_weight": 0.3105}
    ]
}


# 二、核心工具函数（文档模型逻辑实现，含起爆计算）
def calculate_direction_angle(start_pos, end_pos):
    """计算无人机水平运动方向（文档“时空几何关系模型”衍生，格式：北偏东XX度）"""
    start_x, start_y = start_pos[0], start_pos[1]
    end_x, end_y = end_pos[0], end_pos[1]

    delta_x = end_x - start_x  # 东向为正，西向为负
    delta_y = end_y - start_y  # 北向为正，南向为负

    if abs(delta_x) < 1e-6 and abs(delta_y) < 1e-6:
        return "无移动（位置不变）"

    angle_rad = math.atan2(delta_x, delta_y)
    angle_deg = math.degrees(angle_rad)

    # 转换为“北偏东/西”“南偏东/西”表述
    if delta_y >= 0:
        return f"北偏东{round(angle_deg, 1)}度" if delta_x >= 0 else f"北偏西{round(-angle_deg, 1)}度"
    else:
        south_angle = round(abs(angle_deg + 180), 1)
        return f"南偏东{south_angle}度" if delta_x >= 0 else f"南偏西{south_angle}度"


def get_missile_position(missile_id, t):
    """计算t时刻导弹位置（文档公式1.5衍生）"""
    m = missiles[missile_id]
    x0, y0, z0 = m["init_pos"]
    v = m["speed"]
    dx, dy, dz = m["direction"]

    # 导弹最大飞行距离（到真目标的直线距离）
    max_distance = math.sqrt(
        (x0 - real_target[0]) ** 2 + (y0 - real_target[1]) ** 2 + (z0 - real_target[2]) ** 2
    )
    distance = min(v * t, max_distance)  # 避免导弹超过真目标

    return (
        round(x0 + dx * distance, 1),
        round(y0 + dy * distance, 1),
        round(z0 + dz * distance, 1)
    )


def get_uav_flight_path(uav_id, target_missile_id, t):
    """计算t时刻无人机位置（文档“无人机运动模型”衍生）"""
    ux0, uy0, uz0 = uavs[uav_id]["init_pos"]
    mx_t, my_t, _ = get_missile_position(target_missile_id, t)

    dx, dy = mx_t - ux0, my_t - uy0
    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance < 1e-6:
        return (ux0, uy0, uz0)

    # 单位方向向量与实际飞行距离（不超过最大速度×时间）
    dir_x, dir_y = dx / distance, dy / distance
    max_flight = uavs[uav_id]["max_speed"] * t
    move_distance = min(max_flight, distance)

    return (
        round(ux0 + dir_x * move_distance, 1),
        round(uy0 + dir_y * move_distance, 1),
        uz0  # 文档假设无人机水平飞行，高度不变
    )


def calculate_detonation_info(release_time, release_position):
    """计算烟雾弹起爆时间与位置（文档公式1.2、1.3衍生）"""
    detonation_time = round(release_time + smoke_params["detonation_delay"], 1)
    ux, uy, uz = release_position
    # 垂直方向自由下落：z位移=0.5*g*起爆延迟²（g=9.8m/s²）
    z_detonation = round(uz - 0.5 * 9.8 * (smoke_params["detonation_delay"] ** 2), 1)
    return detonation_time, (ux, uy, z_detonation)


# 三、投放计划与指令生成（文档问题五求解逻辑）
def calculate_deployment_details(allocation):
    """计算烟雾弹投放、起爆细节（文档“威胁窗口模型”“投放约束”）"""
    # 1. 计算各导弹威胁窗口期（文档公式5.4、5.5）
    missile_windows = {}
    for mid in missiles:
        m = missiles[mid]
        x0, y0, z0 = m["init_pos"]
        L_total = math.sqrt(
            (x0 - real_target[0]) ** 2 + (y0 - real_target[1]) ** 2 + (z0 - real_target[2]) ** 2
        )
        # 烟雾有效作用半径计算
        fall_height = abs(z0 - sum(u["init_pos"][2] for u in uavs.values()) / len(uavs))
        fall_time = math.sqrt(2 * fall_height / 9.8)
        smoke_total_time = fall_time + 3
        R = m["speed"] * smoke_total_time
        # 窗口期时间范围
        T_in = (L_total - R) / m["speed"]
        T_out = L_total / m["speed"]
        missile_windows[mid] = [round(T_in, 1), round(T_out, 1)]

    # 2. 逐枚计算烟雾弹信息
    deployment_plan = {}
    for mid in allocation:
        deployment_plan[mid] = []
        for assignment in allocation[mid]:
            uid = assignment["uav_id"]
            bomb_count = assignment["assigned_bombs"]

            # 投弹准备截止时间（窗口期前buffer_time秒）
            target_time = missile_windows[mid][0] - smoke_params["buffer_time"]
            # 第一枚弹投放时间（按间隔倒推）
            first_bomb_time = target_time - smoke_params["bomb_interval"] * (bomb_count - 1)

            for i in range(bomb_count):
                bomb_time = round(first_bomb_time + i * smoke_params["bomb_interval"], 1)
                u_pos = get_uav_flight_path(uid, mid, bomb_time)
                m_pos = get_missile_position(mid, bomb_time)
                deto_time, deto_pos = calculate_detonation_info(bomb_time, u_pos)
                # 有效遮蔽时段（以起爆时间为起点，且在导弹窗口期内）
                cover_start = max(deto_time, missile_windows[mid][0])
                cover_end = min(deto_time + smoke_params["single_bomb_duration"], missile_windows[mid][1])

                deployment_plan[mid].append({
                    "bomb_id": f"{uid}-{mid}-{i + 1}",
                    "uav_id": uid,
                    "release_time": bomb_time,
                    "release_position": u_pos,
                    "detonation_time": deto_time,
                    "detonation_position": deto_pos,
                    "missile_position_at_release": m_pos,
                    "coverage_start": cover_start,
                    "coverage_end": cover_end
                })

    return deployment_plan, missile_windows


def generate_uav_mission_commands(deployment_plan):
    """生成无人机运动指令（含运动方向，文档“多机协同策略”）"""
    uav_commands = {uid: {"target_missile": None, "waypoints": [], "main_direction": ""} for uid in uavs}

    # 确定每架无人机的目标导弹
    target_missiles = {}
    for mid in deployment_plan:
        for bomb in deployment_plan[mid]:
            uid = bomb["uav_id"]
            if uid not in target_missiles:
                target_missiles[uid] = mid

    # 生成运动轨迹与方向
    for uid, mid in target_missiles.items():
        uav_commands[uid]["target_missile"] = mid
        start_pos = uavs[uid]["init_pos"]
        bombs = [b for b in deployment_plan[mid] if b["uav_id"] == uid]
        first_bomb_time = min(b["release_time"] for b in bombs)
        last_bomb_time = max(b["release_time"] for b in bombs)
        first_pos = get_uav_flight_path(uid, mid, first_bomb_time)
        last_pos = get_uav_flight_path(uid, mid, last_bomb_time)

        # 计算主方向与分段方向
        main_dir = calculate_direction_angle(start_pos, last_pos)
        first_segment_dir = calculate_direction_angle(start_pos, first_pos)
        last_segment_dir = calculate_direction_angle(first_pos, last_pos)
        uav_commands[uid]["main_direction"] = main_dir

        # 找到第一枚弹的起爆时间
        first_bomb = next((b for b in bombs if b["release_time"] == first_bomb_time), None)
        deto_time = first_bomb["detonation_time"] if first_bomb else first_bomb_time + smoke_params["detonation_delay"]

        # 添加关键节点指令
        uav_commands[uid]["waypoints"].append({
            "time": 0,
            "position": start_pos,
            "action": f"启动，目标导弹{mid}，主运动方向：{main_dir}"
        })
        uav_commands[uid]["waypoints"].append({
            "time": first_bomb_time,
            "position": first_pos,
            "action": f"投放第1枚弹（分段方向：{first_segment_dir}），预计{deto_time}s起爆"
        })
        uav_commands[uid]["waypoints"].append({
            "time": last_bomb_time,
            "position": last_pos,
            "action": f"投放第{len(bombs)}枚弹（分段方向：{last_segment_dir}），完成任务，最后1枚预计{last_bomb_time + smoke_params['detonation_delay']}s起爆"
        })

    return uav_commands


# 四、结果输出（匹配文档“模型求解与分析”格式）
def print_deployment_results(deployment_plan, uav_commands):
    print("===== 《25数模(2).docx》问题五：多机多弹烟雾弹投放策略结果 =====")

    # 1. 无人机运动指令详情
    print("\n【一、无人机运动指令（含方向）】")
    for uid, cmd in uav_commands.items():
        if cmd["target_missile"]:
            print(f"\n无人机{uid}：")
            print(f"  - 目标导弹：{cmd['target_missile']}")
            print(f"  - 主运动方向：{cmd['main_direction']}")
            print(f"  - 关键节点：")
            for wp in cmd["waypoints"]:
                print(f"    · t={wp['time']}s：位置{wp['position']} | {wp['action']}")

    # 2. 烟雾弹投放与起爆详情
    print("\n【二、烟雾弹投放与起爆详情】")
    total_bombs = 0
    for mid in deployment_plan:
        print(f"\n针对导弹{mid}：")
        uav_bomb_cnt = {}
        for bomb in deployment_plan[mid]:
            uid = bomb["uav_id"]
            uav_bomb_cnt[uid] = uav_bomb_cnt.get(uid, 0) + 1
            total_bombs += 1
            print(f"  - 烟雾弹{bomb['bomb_id']}：")
            print(f"    · 投放时间：{bomb['release_time']}s | 投放位置：{bomb['release_position']}（x,y,z）")
            print(f"    · 起爆时间：{bomb['detonation_time']}s | 起爆位置：{bomb['detonation_position']}（x,y,z）")
            print(f"    · 投放时导弹位置：{bomb['missile_position_at_release']}（x,y,z）")
            print(f"    · 有效遮蔽时段：{bomb['coverage_start']}s ~ {bomb['coverage_end']}s")
        print(f"  - 投弹汇总：{[f'{uid}投{cnt}枚' for uid, cnt in uav_bomb_cnt.items()]}")

    # 3. 核心统计
    print(f"\n【三、核心统计】")
    print(f"  - 总投放烟雾弹数量：{total_bombs}枚（符合“每架无人机至多3枚”约束）")
    print(f"  - 覆盖导弹数量：{len(deployment_plan)}枚（M1、M2、M3）")
    print(f"  - 参与无人机数量：{len([uid for uid in uav_commands if uav_commands[uid]['target_missile']])}架")
    print(f"  - 烟雾弹通用参数：起爆延迟{smoke_params['detonation_delay']}s，单枚有效时长{smoke_params['single_bomb_duration']}s")


# 五、执行主逻辑（文档问题五求解流程）
if __name__ == "__main__":
    deployment_plan, missile_windows = calculate_deployment_details(optimal_allocation)
    uav_commands = generate_uav_mission_commands(deployment_plan)
    print_deployment_results(deployment_plan, uav_commands)