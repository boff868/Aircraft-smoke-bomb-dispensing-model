import math

# 基于最优分配的烟雾弹投放计划（简化版，包含核心参数）
deployment_plan = {
    "M3": [
        {"bomb_id": "FY4-M3-1", "release_time": 44.3, "coverage_start": 47.3, "coverage_end": 57.3},
        {"bomb_id": "FY4-M3-2", "release_time": 45.3, "coverage_start": 47.3, "coverage_end": 58.3},
        {"bomb_id": "FY4-M3-3", "release_time": 46.3, "coverage_start": 47.3, "coverage_end": 59.3},
        {"bomb_id": "FY3-M3-1", "release_time": 45.0, "coverage_start": 47.3, "coverage_end": 58.0},
        {"bomb_id": "FY3-M3-2", "release_time": 46.0, "coverage_start": 47.3, "coverage_end": 59.0}
    ],
    "M2": [
        {"bomb_id": "FY5-M2-1", "release_time": 45.8, "coverage_start": 48.8, "coverage_end": 58.8},
        {"bomb_id": "FY5-M2-2", "release_time": 46.8, "coverage_start": 48.8, "coverage_end": 59.8},
        {"bomb_id": "FY5-M2-3", "release_time": 47.8, "coverage_start": 48.8, "coverage_end": 60.8},
        {"bomb_id": "FY2-M2-1", "release_time": 46.8, "coverage_start": 48.8, "coverage_end": 59.8},
        {"bomb_id": "FY2-M2-2", "release_time": 47.8, "coverage_start": 48.8, "coverage_end": 60.8},
        {"bomb_id": "FY3-M2-1", "release_time": 47.8, "coverage_start": 48.8, "coverage_end": 60.8}
    ],
    "M1": [
        {"bomb_id": "FY1-M1-1", "release_time": 50.9, "coverage_start": 52.9, "coverage_end": 62.9},
        {"bomb_id": "FY1-M1-2", "release_time": 51.9, "coverage_start": 52.9, "coverage_end": 63.9},
        {"bomb_id": "FY1-M1-3", "release_time": 52.9, "coverage_start": 52.9, "coverage_end": 64.9},
        {"bomb_id": "FY2-M1-1", "release_time": 51.9, "coverage_start": 52.9, "coverage_end": 63.9}
    ]
}

# 导弹窗口期
missile_windows = {
    "M3": [47.3, 60.4],
    "M2": [48.8, 63.7],
    "M1": [52.9, 67.0]
}


def calculate_effective_coverage(missile_id, bomb_plan):
    """计算单枚导弹的有效遮盖时间（去重叠）"""
    # 提取该导弹的所有烟雾弹遮盖时段
    time_segments = []
    for bomb in bomb_plan:
        if bomb["coverage_start"] < bomb["coverage_end"]:  # 过滤无效时段
            time_segments.append([bomb["coverage_start"], bomb["coverage_end"]])

    # 按开始时间排序
    time_segments.sort(key=lambda x: x[0])
    if not time_segments:
        return 0.0, []

    # 合并重叠时段
    merged_segments = [time_segments[0]]
    for current in time_segments[1:]:
        last = merged_segments[-1]
        if current[0] <= last[1]:  # 重叠或相邻，合并
            merged_segments[-1] = [last[0], max(last[1], current[1])]
        else:  # 无重叠，新增
            merged_segments.append(current)

    # 计算总有效时间
    total_time = sum(end - start for start, end in merged_segments)
    return round(total_time, 1), merged_segments


def calculate_total_coverage(all_plans, windows):
    """计算所有导弹的总有效遮盖时间"""
    total_coverage = 0.0
    detail = {}

    for missile_id, plan in all_plans.items():
        effective_time, segments = calculate_effective_coverage(missile_id, plan)
        window_length = round(windows[missile_id][1] - windows[missile_id][0], 1)

        detail[missile_id] = {
            "window": windows[missile_id],
            "window_length": window_length,
            "effective_time": effective_time,
            "coverage_ratio": round((effective_time / window_length) * 100, 1) if window_length > 0 else 0,
            "merged_segments": segments
        }

        total_coverage += effective_time

    return round(total_coverage, 1), detail

def print_coverage_result(total_time, detail):
    print("===== 烟雾弹有效遮盖时间计算结果 =====")
    print("核心逻辑：合并重叠时段，仅计算实际有效覆盖时间\n")

    # 各导弹详情
    for missile_id, info in detail.items():
        print(f"【导弹{missile_id}】")
        print(f"   威胁窗口期：{info['window'][0]}s ~ {info['window'][1]}s（时长{info['window_length']}s）")
        print(f"   有效遮盖时段：{['%0.1fs~%0.1fs' % (s, e) for s, e in info['merged_segments']]}")
        print(f"   有效遮盖时间：{info['effective_time']}s")
        print(f"   窗口期覆盖比例：{info['coverage_ratio']}%")
        print()

    # 总遮盖时间
    print(f"所有导弹总有效遮盖时间：{total_time}s")
    print(f"注：已扣除所有重叠时段，反映实际防护效果")



if __name__ == "__main__":
    total_coverage, coverage_detail = calculate_total_coverage(deployment_plan, missile_windows)
    print_coverage_result(total_coverage, coverage_detail)
# 导弹参数
missiles = {
    "M1": {"init_pos": (20000, 0, 2000), "speed": 300, "direction": (-1, 0, 0)},
    "M2": {"init_pos": (19000, 600, 2100), "speed": 300, "direction": (-0.98, -0.06, 0)},
    "M3": {"init_pos": (18000, -600, 1900), "speed": 300, "direction": (-0.97, 0.07, 0)}
}

# 无人机参数
uavs = {
    "FY1": {"init_pos": (17800, 0, 1800), "max_speed": 140, "max_bombs": 3},
    "FY2": {"init_pos": (12000, 1400, 1400), "max_speed": 140, "max_bombs": 3},
    "FY3": {"init_pos": (6000, -3000, 700), "max_speed": 140, "max_bombs": 3},
    "FY4": {"init_pos": (11000, 2000, 1800), "max_speed": 140, "max_bombs": 3},
    "FY5": {"init_pos": (13000, -2000, 1300), "max_speed": 140, "max_bombs": 3}
}

# 真目标与烟幕参数
real_target = (0, 200, 5)
smoke_params = {
    "buffer_time": 2,  # 投弹缓冲时间
    "single_bomb_duration": 10,  # 单枚弹有效时间
    "bomb_interval": 1  # 投弹间隔
}

# 基于t=10s决策的最优分配方案（来自动态分配结果）
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


# -------------------------- 动态位置计算函数 --------------------------
def get_missile_position(missile_id, t):
    """计算t时刻导弹的实时位置"""
    m = missiles[missile_id]
    x0, y0, z0 = m["init_pos"]
    v = m["speed"]
    dx, dy, dz = m["direction"]

    max_distance = math.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
    distance = min(v * t, max_distance)

    return (
        round(x0 + dx * distance, 1),
        round(y0 + dy * distance, 1),
        round(z0 + dz * distance, 1)
    )


def get_uav_flight_path(uav_id, target_missile_id, t):
    """计算无人机在t时刻的位置（从0时刻开始向目标导弹移动）"""
    ux0, uy0, uz0 = uavs[uav_id]["init_pos"]
    mx_t, my_t, _ = get_missile_position(target_missile_id, t)

    # 计算无人机到t时刻导弹位置的方向
    dx, dy = mx_t - ux0, my_t - uy0
    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance < 1e-6:
        return (ux0, uy0, uz0)

    # 单位方向向量
    dir_x, dir_y = dx / distance, dy / distance

    # 最大飞行距离（速度×时间）
    max_flight = uavs[uav_id]["max_speed"] * t
    move_distance = min(max_flight, distance)

    return (
        round(ux0 + dir_x * move_distance, 1),
        round(uy0 + dir_y * move_distance, 1),
        uz0  # 高度不变
    )


# -------------------------- 投放时间与地点计算 --------------------------
def calculate_deployment_details(allocation):
    """计算每枚烟雾弹的投放时间、地点及无人机运动轨迹"""
    # 1. 先计算各导弹窗口期
    missile_windows = {}
    for mid in missiles:
        m = missiles[mid]
        x0, y0, z0 = m["init_pos"]
        L_total = math.sqrt(
            (x0 - real_target[0]) ** 2 + (y0 - real_target[1]) ** 2 + (z0 - real_target[2]) ** 2
        )
        fall_height = abs(z0 - sum(u["init_pos"][2] for u in uavs.values()) / len(uavs))
        fall_time = math.sqrt(2 * fall_height / 9.8)
        smoke_total_time = fall_time + 3  # 下落时间+延迟时间
        R = m["speed"] * smoke_total_time
        T_in = (L_total - R) / m["speed"]
        T_out = L_total / m["speed"]
        missile_windows[mid] = [round(T_in, 1), round(T_out, 1)]

    deployment_plan = {}

    # 2. 计算每架无人机的任务
    for mid in allocation:
        deployment_plan[mid] = []
        for assignment in allocation[mid]:
            uid = assignment["uav_id"]
            bomb_count = assignment["assigned_bombs"]

            # 计算无人机到达投弹点的时间
            # 投弹点：导弹飞行路径上的最佳拦截点（窗口期开始前buffer_time）
            target_time = missile_windows[mid][0] - smoke_params["buffer_time"]
            mx_target, my_target, mz_target = get_missile_position(mid, target_time)

            # 无人机需要在target_time前到达投弹点
            ux0, uy0, _ = uavs[uid]["init_pos"]
            distance_needed = math.sqrt((mx_target - ux0) ** 2 + (my_target - uy0) ** 2)
            flight_time_needed = distance_needed / uavs[uid]["max_speed"]

            # 第一枚弹投放时间（到达后立即投放）
            first_bomb_time = target_time - smoke_params["bomb_interval"] * (bomb_count - 1)

            # 每枚弹的具体投放信息
            for i in range(bomb_count):
                bomb_time = round(first_bomb_time + i * smoke_params["bomb_interval"], 1)
                ux, uy, uz = get_uav_flight_path(uid, mid, bomb_time)
                mx, my, mz = get_missile_position(mid, bomb_time)

                deployment_plan[mid].append({
                    "bomb_id": f"{uid}-{mid}-{i + 1}",
                    "uav_id": uid,
                    "missile_id": mid,
                    "release_time": bomb_time,  # 投放时间（s）
                    "release_position": (ux, uy, uz),  # 投放地点（m）
                    "missile_position_at_release": (mx, my, mz),  # 投放时导弹位置
                    "coverage_start": max(bomb_time, missile_windows[mid][0]),
                    "coverage_end": min(bomb_time + smoke_params["single_bomb_duration"], missile_windows[mid][1])
                })

    return deployment_plan, missile_windows

def generate_uav_mission_commands(deployment_plan):
    """生成各无人机的运动指令（从0时刻开始）"""
    uav_commands = {uid: {"target_missile": None, "waypoints": []} for uid in uavs}

    # 确定每架无人机的目标导弹
    target_missiles = {}
    for mid in deployment_plan:
        for bomb in deployment_plan[mid]:
            uid = bomb["uav_id"]
            if uid not in target_missiles:
                target_missiles[uid] = mid

    # 生成运动指令
    for uid, mid in target_missiles.items():
        uav_commands[uid]["target_missile"] = mid
        # 0时刻位置
        start_pos = uavs[uid]["init_pos"]
        uav_commands[uid]["waypoints"].append({
            "time": 0,
            "position": start_pos,
            "action": "开始向导弹{}移动".format(mid)
        })

        # 投放第一枚弹前的位置
        bombs = [b for b in deployment_plan[mid] if b["uav_id"] == uid]
        first_bomb_time = min(b["release_time"] for b in bombs)
        first_pos = get_uav_flight_path(uid, mid, first_bomb_time)
        uav_commands[uid]["waypoints"].append({
            "time": first_bomb_time,
            "position": first_pos,
            "action": "投放第1枚弹"
        })

        # 最后一枚弹投放后的位置
        last_bomb_time = max(b["release_time"] for b in bombs)
        last_pos = get_uav_flight_path(uid, mid, last_bomb_time)
        uav_commands[uid]["waypoints"].append({
            "time": last_bomb_time,
            "position": first_pos,
            "action": "完成所有投弹任务"
        })

    return uav_commands


def print_deployment_details(deployment_plan, uav_commands):
    print("===== 烟雾弹投放详细计划 =====")

    # 1. 无人机运动指令
    print("\n【无人机运动指令（从0时刻开始）】")
    for uid, cmd in uav_commands.items():
        if cmd["target_missile"]:
            print(f"\n无人机{uid}：")
            print(f"   目标导弹：{cmd['target_missile']}")
            print(f"   运动轨迹：")
            for wp in cmd["waypoints"]:
                print(f"      t={wp['time']}s: 位置{wp['position']}，动作：{wp['action']}")

    # 2. 烟雾弹投放详情
    print("\n\n【烟雾弹投放详情】")
    total_bombs = 0
    for mid in deployment_plan:
        print(f"\n针对导弹{mid}：")
        for bomb in deployment_plan[mid]:
            total_bombs += 1
            print(f"   烟雾弹{bomb['bomb_id']}：")
            print(f"      投放时间：{bomb['release_time']}s")
            print(f"      投放位置：{bomb['release_position']}")
            print(f"      投放时导弹位置：{bomb['missile_position_at_release']}")
            print(f"      有效遮盖时段：{bomb['coverage_start']}s ~ {bomb['coverage_end']}s")

    print(f"\n总投放烟雾弹数量：{total_bombs}枚")


if __name__ == "__main__":
    # 计算投放详情
    deployment_plan, _ = calculate_deployment_details(optimal_allocation)
    # 生成无人机运动指令
    uav_commands = generate_uav_mission_commands(deployment_plan)
    # 输出结果
    print_deployment_details(deployment_plan, uav_commands)