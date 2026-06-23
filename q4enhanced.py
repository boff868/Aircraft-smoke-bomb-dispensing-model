from CheckFunc import check
import numpy as np
import logging
from tqdm import tqdm
import pandas as pd
from scipy.stats import qmc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
USE_PENALTY = True




PARAM_BOUNDS = {
    'angle': [0, 360], 'speed': [70, 140],
    'drop_time': [1.0, 50.0], 'delay_time': [1.0, 15.0]
}
LOWER_BOUNDS = np.array([b[0] for b in PARAM_BOUNDS.values()])
UPPER_BOUNDS = np.array([b[1] for b in PARAM_BOUNDS.values()])
PARAM_RANGE = UPPER_BOUNDS - LOWER_BOUNDS


def calculate_union_duration(intervals):
    if not any(intervals): return 0
    valid_intervals = sorted([iv for iv in intervals if iv and iv[0] < iv[1]], key=lambda x: x[0])
    if not valid_intervals: return 0
    merged = []
    for start, end in valid_intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)



def objective_function(strategy, drone_pos, missile_pos):
    result = check(strategy[0], strategy[1], strategy[2], strategy[3],
                       drone_pos[0], drone_pos[1], drone_pos[2],
                       missile_pos[0], missile_pos[1], missile_pos[2])
    duration = result[0]
    if duration > 0: return duration, result
    if USE_PENALTY:
        normalized_strategy = (np.array(strategy) - LOWER_BOUNDS) / PARAM_RANGE
        distance_from_center = np.linalg.norm(normalized_strategy - 0.5)
        return -100.0 - distance_from_center, result
    return 0.0, result



def _evaluate_and_collect(strategies, drone_name, drone_pos, missile_pos, desc=""):
    candidates = []
    max_duration_so_far = 0.0
    total_strategies = len(strategies)
    if total_strategies == 0: return []

    checkpoint_interval = max(1, total_strategies // 20)  # 5% 间隔

    with tqdm(strategies, desc=f"评估 {drone_name} ({desc})") as pbar:
        for i, strategy in enumerate(pbar):
            fitness, result = objective_function(np.array(strategy), drone_pos, missile_pos)
            if fitness > 0:
                duration, start_time, end_time = result
                candidates.append({'drone': drone_name, 'strategy': strategy, 'interval': (start_time, end_time),
                                   'duration': duration})
                if duration > max_duration_so_far:
                    max_duration_so_far = duration

            if (i + 1) % checkpoint_interval == 0:
                pbar.set_description(f"评估 {drone_name} ({desc}) (当前最优: {max_duration_so_far:.3f}s)")
    return candidates


class PSO_Optimizer:
    def __init__(self, drone_name, drone_pos, missile_pos, num_particles, max_iter, initial_guess=None, w_max=0.5,
                 w_min=0.1, c1=1.5, c2=1.5):
        self.drone_name, self.drone_pos, self.missile_pos = drone_name, drone_pos, missile_pos
        self.num_particles, self.max_iter = num_particles, max_iter
        self.w_max, self.w_min, self.c1, self.c2 = w_max, w_min, c1, c2
        self.particles_pos = np.random.rand(num_particles, 4) * PARAM_RANGE + LOWER_BOUNDS
        self.particles_vel = (np.random.rand(num_particles, 4) - 0.5) * PARAM_RANGE * 0.05
        self.pbest_pos = self.particles_pos.copy()
        self.pbest_val = np.full(num_particles, -np.inf)
        self.pbest_results = [[0, 0, 0]] * num_particles
        self.gbest_pos, self.gbest_val, self.gbest_result = self.pbest_pos[0].copy(), -np.inf, [0, 0, 0]
        if initial_guess is not None:
            self.particles_pos[0] = np.array(initial_guess)
            fitness, result = objective_function(np.array(initial_guess), self.drone_pos, self.missile_pos)
            self.pbest_pos[0], self.pbest_val[0], self.pbest_results[0] = np.array(initial_guess), fitness, result
            if fitness > self.gbest_val:
                self.gbest_val, self.gbest_pos, self.gbest_result = fitness, np.array(initial_guess), result

    def optimize(self):
        logging.info(f"开始为 {self.drone_name} 运行粒子群优化 (PSO)...")
        checkpoint_interval = max(1, self.max_iter // 20)  # 5% 间隔

        with tqdm(range(self.max_iter), desc=f"PSO {self.drone_name}") as pbar:
            for i in pbar:
                w = self.w_max - (self.w_max - self.w_min) * (i / self.max_iter)
                for j in range(self.num_particles):
                    strategy = self.particles_pos[j]
                    fitness, result = objective_function(strategy, self.drone_pos, self.missile_pos)
                    if fitness > self.pbest_val[j]:
                        self.pbest_val[j], self.pbest_pos[j], self.pbest_results[j] = fitness, strategy, result
                    if fitness > self.gbest_val:
                        self.gbest_val, self.gbest_pos, self.gbest_result = fitness, strategy, result

                for j in range(self.num_particles):
                    r1, r2 = np.random.rand(2)
                    cognitive_vel = self.c1 * r1 * (self.pbest_pos[j] - self.particles_pos[j])
                    social_vel = self.c2 * r2 * (self.gbest_pos - self.particles_pos[j])
                    self.particles_vel[j] = w * self.particles_vel[j] + cognitive_vel + social_vel
                    self.particles_pos[j] = np.clip(self.particles_pos[j] + self.particles_vel[j], LOWER_BOUNDS,
                                                    UPPER_BOUNDS)

                if (i + 1) % checkpoint_interval == 0 or i == self.max_iter - 1:
                    current_best_duration = max(0, self.gbest_val)
                    pbar.set_description(f"PSO {self.drone_name} (当前最优: {current_best_duration:.3f}s)")

        return [
            {'drone': self.drone_name, 'strategy': self.pbest_pos[i], 'interval': (res[1], res[2]), 'duration': res[0]}
            for i, res in enumerate(self.pbest_results) if self.pbest_val[i] > 0]


def generate_candidates_pso(drone_name, drone_pos, missile_pos, num_candidates_to_find, initial_guess=None):
    optimizer = PSO_Optimizer(drone_name, drone_pos, missile_pos, 50, 30, initial_guess)
    candidates = optimizer.optimize()
    if not candidates: logging.warning(f"PSO 未能为 {drone_name} 找到任何有效的候选方案。")
    return sorted(candidates, key=lambda x: x['duration'], reverse=True)[:num_candidates_to_find]



def find_best_combination(candidate_lists):
    logging.info("开始枚举所有候选组合以寻找最优协同策略...")
    best_combination, max_union_duration = None, 0.0
    if not all(candidate_lists):
        logging.error("错误: 至少有一个无人机的候选列表为空，无法进行组合。")
        return None

    cand1_list, cand2_list, cand3_list = candidate_lists
    total_combinations = len(cand1_list) * len(cand2_list) * len(cand3_list)

    with tqdm(total=total_combinations, desc="组合选择") as pbar:
        for c1 in cand1_list:
            for c2 in cand2_list:
                for c3 in cand3_list:
                    intervals = [c1.get('interval'), c2.get('interval'), c3.get('interval')]
                    current_union_duration = calculate_union_duration(intervals)
                    if current_union_duration > max_union_duration:
                        max_union_duration = current_union_duration
                        best_combination = {'FY1': c1, 'FY2': c2, 'FY3': c3, 'total_duration': current_union_duration}
                        # 更新进度条描述
                        pbar.set_description(f"组合选择 (当前最优: {max_union_duration:.3f}s)")
                    pbar.update(1)
    return best_combination


def verify_optimization_effect(original_combo, optimized_combo, alpha=0.05):
    logging.info("开始局部优化效果验证...")
    result = {
        "optimization_effective": False,
        "duration_increase": 0.0,
        "increase_ratio": 0.0,
        "original_total": original_combo["total_duration"],
        "optimized_total": optimized_combo["total_duration"],
        "key_findings": []
    }

    # 1. 计算时长提升
    result["duration_increase"] = result["optimized_total"] - result["original_total"]
    result["increase_ratio"] = (result["duration_increase"] / result["original_total"]) * 100 if result["original_total"] > 0 else 0.0

    # 2. 验证参数是否符合文档约束（角度0-360°、速度70-140m/s等）
    params_valid = True
    bounds_check = {"angle": (0, 360), "speed": (70, 140), "drop_time": (1.0, 50.0), "delay_time": (1.0, 15.0)}
    for drone in ["FY1", "FY2", "FY3"]:
        opt_strat = optimized_combo[drone]["strategy"]
        for i, param_name in enumerate(["angle", "speed", "drop_time", "delay_time"]):
            min_b, max_b = bounds_check[param_name]
            if not (min_b <= opt_strat[i] <= max_b):
                params_valid = False
                result["key_findings"].append(f"无人机{drone}的{param_name}（{opt_strat[i]:.2f}）超出约束[{min_b},{max_b}]")

    # 3. 验证参数稳定性（小范围微调，波动不超过20%）
    stable = True
    param_fluctuation = {}
    for drone in ["FY1", "FY2", "FY3"]:
        orig_strat = original_combo[drone]["strategy"]
        opt_strat = optimized_combo[drone]["strategy"]
        for i, param_name in enumerate(["angle", "speed", "drop_time", "delay_time"]):
            if orig_strat[i] != 0:
                fluct = abs((opt_strat[i] - orig_strat[i]) / orig_strat[i]) * 100
            else:
                fluct = abs(opt_strat[i] - orig_strat[i])
            if fluct > 20:
                stable = False
                param_fluctuation[drone] = param_fluctuation.get(drone, {})
                param_fluctuation[drone][param_name] = f"{fluct:.1f}%"
    if param_fluctuation:
        result["key_findings"].append(f"参数波动超20%（不稳定）：{param_fluctuation}")

    # 4. 有效性判定（时长提升≥0.1s+参数有效+稳定）
    absolute_effective = result["duration_increase"] >= 0.1  # 排除计算误差
    if absolute_effective and params_valid and stable:
        result["optimization_effective"] = True
        result["key_findings"].insert(0, f"优化有效！时长从{result['original_total']:.3f}s→{result['optimized_total']:.3f}s，提升{result['duration_increase']:.3f}s（{result['increase_ratio']:.1f}%）")
    else:
        if not absolute_effective:
            result["key_findings"].insert(0, f"优化无效：时长提升{result['duration_increase']:.3f}s（<0.1s，视为误差）")
        if not params_valid:
            result["key_findings"].insert(0, "优化无效：参数违反文档约束")
        if not stable:
            result["key_findings"].insert(0, "优化无效：参数波动过大")

    # 打印验证报告
    print("\n" + "="*70)
    print("局部优化效果验证报告")
    print("="*70)
    print(f"优化前总遮蔽时长：{result['original_total']:.3f}s")
    print(f"优化后总遮蔽时长：{result['optimized_total']:.3f}s")
    print(f"时长提升：{result['duration_increase']:.3f}s（{result['increase_ratio']:.1f}%）")
    print(f"优化有效性：{'有效' if result['optimization_effective'] else '无效'}")
    print("\n关键结论：")
    for idx, finding in enumerate(result["key_findings"], 1):
        print(f"  {idx}. {finding}")
    print("="*70)

    return result


def local_optimization(best_combo, lr=0.01, max_iter=20, tolerance=1e-4):
    logging.info("开始组合后局部优化...")
    # 1. 提取初始参数（12维：3架无人机×4个参数）
    def extract_params(combo):
        params = []
        for drone in ['FY1', 'FY2', 'FY3']:
            params.extend(combo[drone]['strategy'])
        return np.array(params, dtype=np.float64)

    # 2. 计算参数对应的总遮蔽时长（调用原check函数）
    def calculate_total_duration(params):
        positions = {'FY1': (17800, 0, 1800), 'FY2': (12000, 1400, 1400), 'FY3': (6000, -3000, 700)}
        pos_m1 = (20000, 0, 2000)
        intervals = []
        # 拆分参数到各无人机
        fy1_params = params[0:4]
        fy2_params = params[4:8]
        fy3_params = params[8:12]
        # 计算每架无人机的遮蔽区间
        for params, pos in zip([fy1_params, fy2_params, fy3_params], positions.values()):
            duration, t_start, t_end = check(
                params[0], params[1], params[2], params[3],
                pos[0], pos[1], pos[2],
                pos_m1[0], pos_m1[1], pos_m1[2]
            )
            intervals.append((t_start, t_end) if duration > 0 else (0, 0))
        return calculate_union_duration(intervals)

    # 3. 参数边界（对齐原PARAM_BOUNDS，确保不超约束）
    bounds = [
        (0, 360), (70, 140), (1.0, 50.0), (1.0, 15.0),  # FY1：角度、速度、投放时间、延时
        (0, 360), (70, 140), (1.0, 50.0), (1.0, 15.0),  # FY2
        (0, 360), (70, 140), (1.0, 50.0), (1.0, 15.0)   # FY3
    ]

    # 4. 迭代优化（完整执行max_iter次，不提前返回）
    current_params = extract_params(best_combo)
    current_best_duration = best_combo['total_duration']
    best_params = current_params.copy()

    with tqdm(range(max_iter), desc="局部优化迭代") as pbar:
        for _ in pbar:
            # 随机选择1个参数双向微调（避免维度爆炸）
            param_idx = np.random.randint(0, 12)
            for delta in [lr, -lr]:
                new_params = current_params.copy()
                new_params[param_idx] += delta
                # 确保参数在边界内
                new_params[param_idx] = np.clip(new_params[param_idx], bounds[param_idx][0], bounds[param_idx][1])
                # 计算新参数的总时长
                new_duration = calculate_total_duration(new_params)
                # 更新最优参数（需满足时长提升超阈值）
                if new_duration > current_best_duration + tolerance:
                    current_best_duration = new_duration
                    best_params = new_params.copy()
                    pbar.set_description(f"局部优化迭代（当前最优: {current_best_duration:.3f}s）")
            # 更新当前参数为最优参数
            current_params = best_params

    # 5. 重构优化后的组合字典
    optimized_combo = best_combo.copy()
    positions = {'FY1': (17800, 0, 1800), 'FY2': (12000, 1400, 1400), 'FY3': (6000, -3000, 700)}
    pos_m1 = (20000, 0, 2000)
    # 拆分参数并更新每架无人机策略、区间、时长
    fy1_opt = best_params[0:4]
    fy2_opt = best_params[4:8]
    fy3_opt = best_params[8:12]
    for drone, params, pos in zip(['FY1', 'FY2', 'FY3'], [fy1_opt, fy2_opt, fy3_opt], positions.values()):
        duration, t_start, t_end = check(
            params[0], params[1], params[2], params[3],
            pos[0], pos[1], pos[2],
            pos_m1[0], pos_m1[1], pos_m1[2]
        )
        optimized_combo[drone]['strategy'] = params
        optimized_combo[drone]['interval'] = (t_start, t_end)
        optimized_combo[drone]['duration'] = duration
    optimized_combo['total_duration'] = current_best_duration

    # 6. 调用效果验证模块
    verify_result = verify_optimization_effect(best_combo, optimized_combo)
    optimized_combo["optimization_verification"] = verify_result

    logging.info("局部优化及效果验证完成")
    return optimized_combo

def main():
    # --- 模型参数 ---（原代码不变，此处省略以避免重复）
    POSITIONS = {'FY1': (17800, 0, 1800), 'FY2': (12000, 1400, 1400), 'FY3': (6000, -3000, 700)}
    POS_M1 = (20000, 0, 2000)

    INITIAL_GUESSES = {
        'FY1': [6.747627627919481, 71.69288375357752, 0.3056, 0.9054],
        'FY2': [286.15913456920737, 98.7487963287457, 8.7730, 5.5698],
        'FY3': [99.21882113404057,88.67913077832478,29.749794019785924,5.405090225915116]
    }

    SAMPLING_METHOD = "PSO"
    N_CANDIDATES = 50

  #  步骤 1: 候选生成
    all_candidates = []
    for name, pos in POSITIONS.items():
        current_guess = INITIAL_GUESSES.get(name, None)
        if SAMPLING_METHOD == "PSO":
            candidates = generate_candidates_pso(name, pos, POS_M1, N_CANDIDATES, initial_guess=current_guess)
        else:
            raise ValueError("当前配置仅支持PSO，请修改SAMPLING_METHOD或添加其他算法实现。")
        all_candidates.append(candidates)
#步骤 2: 组合选择
    best_combo = find_best_combination(all_candidates)
    if not best_combo:
        logging.error("未能找到任何有效的三机协同策略。")
        return

    # 新增步骤: 局部优化
    print("\n" + "="*85)
    print("开始局部优化最优组合...")
    print("="*85)
    best_combo_optimized = local_optimization(best_combo)
    # 更新为优化后的组合（后续输出使用优化后结果）
    best_combo = best_combo_optimized

    # 步骤 3: 输出与分析
    print("      " + " = "*85)
    print(" " * 28 + "最优三机协同策略分析结果（含局部优化）")
    print(f"(候选生成: {SAMPLING_METHOD}, 初始猜测: 自定义, 惩罚机制: {'启用' if USE_PENALTY else '禁用'})".center(85))
    print("=" * 85)
    print(f"      [+]最大有效遮蔽总时长(并集): {best_combo['total_duration']: .3f}秒")
    print("-" * 85)

    results_for_df, total_individual_duration = [], 0
    for drone_name in ['FY1', 'FY2', 'FY3']:
        details = best_combo[drone_name]
        strategy, duration, interval = details['strategy'], details['duration'], details.get('interval', (0, 0))
        total_individual_duration += duration

        print(f"  - 无人机: {drone_name:<3s} | "
              f"策略(角,速,投,引): ({strategy[0]:.4f}, {strategy[1]:.4f}, {strategy[2]:.4f}, {strategy[3]:.4f}) | "
              f"独立贡献: {duration:>6.3f}s | "
              f"覆盖时段: [{interval[0]:.3f}s, {interval[1]:.3f}s]")
        results_for_df.append({'无人机编号': drone_name, '飞行方向(角度)': strategy[0], '飞行速度(m/s)': strategy[1],
                               '投放时间(s)': strategy[2], '引信延时(s)': strategy[3]})

    print("-" * 85)
    print("--- 协同效率分析 ---")
    overlap_waste = total_individual_duration - best_combo['total_duration']
    waste_ratio = (overlap_waste / total_individual_duration) * 100 if total_individual_duration > 0 else 0

    print(f"  - 协同分析: 独立总和 {total_individual_duration:.3f}s | "
          f"重叠损失 {overlap_waste:.3f}s | "
          f"损失率 {waste_ratio:.1f}%")

    if waste_ratio < 10:
        print("  - 评价: 协同效率高，各机时间窗口分配良好。")
    else:
        print("  - 评价: 协同效率中等，存在一定的重叠，但策略有效。")
    print("=" * 85)
    print("\n" + "=" * 85)
    print("开始策略稳定性评估（计算置信区间）...")
    print("=" * 85)
    stability_result = evaluate_strategy_stability(best_combo, n_trials=30, confidence=0.95)
    # 将置信区间结果添加到组合字典，便于后续保存
    best_combo["stability_analysis"] = stability_result

    results_for_df.extend([
        {'无人机编号': '策略稳定性', '飞行方向(角度)': '-', '飞行速度(m/s)': '-',
         '投放时间(s)': '-', '引信延时(s)': '-',
         '独立遮蔽时长(s)': f"{stability_result['confidence_level'] * 100}%置信区间：[{stability_result['ci_lower']:.3f}s, {stability_result['ci_upper']:.3f}s]"},
        {'无人机编号': '', '飞行方向(角度)': '-', '飞行速度(m/s)': '-',
         '投放时间(s)': '-', '引信延时(s)': '-',
         '独立遮蔽时长(s)': f"均值：{stability_result['mean_total_duration']:.3f}s | 变异系数：{stability_result['cv_percent']:.1f}%"}
    ])

    try:
        df = pd.DataFrame(results_for_df)
        df.to_excel('result2.xlsx', index=False, engine='openpyxl')
        logging.info(f"    含置信区间的结果已成功保存到文件: result2.xlsx    ")
    except Exception as e:
        logging.error(f"    保存结果到Excel文件时出错: {e}    ")

    try:
        df = pd.DataFrame(results_for_df)
        df.to_excel('result2.xlsx', index=False, engine='openpyxl')
        logging.info(f"    优化后的结果已成功保存到文件: result2.xlsx    ")
    except Exception as e:
        logging.error(f"    保存结果到Excel文件时出错: {e}    ")


def calculate_confidence_interval(data, confidence=0.95):
    import scipy.stats as stats
    n = len(data)
    if n < 2:
        logging.warning("样本量不足2，无法计算置信区间，返回均值与数据本身")
        return (np.mean(data), min(data), max(data), np.std(data, ddof=1) if n > 1 else 0)

    # 计算均值、标准差、标准误
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)  # 样本标准差
    se_val = std_val / np.sqrt(n)  # 标准误
    # 计算t临界值（双侧检验）
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    # 置信区间上下限
    ci_lower = mean_val - t_critical * se_val
    ci_upper = mean_val + t_critical * se_val
    return (mean_val, ci_lower, ci_upper, std_val)


def evaluate_strategy_stability(best_combo, n_trials=30, confidence=0.95):
    logging.info(f"开始策略稳定性评估（{n_trials}次重复实验，{confidence * 100}%置信水平）...")
    trial_durations = []
    positions = {'FY1': (17800, 0, 1800), 'FY2': (12000, 1400, 1400), 'FY3': (6000, -3000, 700)}
    pos_m1 = (20000, 0, 2000)

    with tqdm(range(n_trials), desc="稳定性评估（重复实验）") as pbar:
        for _ in pbar:
            # 每次实验重新计算总遮蔽时长（模拟数值波动）
            intervals = []
            for drone in ['FY1', 'FY2', 'FY3']:
                strat = best_combo[drone]['strategy']
                duration, t_start, t_end = check(
                    strat[0], strat[1], strat[2], strat[3],
                    positions[drone][0], positions[drone][1], positions[drone][2],
                    pos_m1[0], pos_m1[1], pos_m1[2]
                )
                intervals.append((t_start, t_end) if duration > 0 else (0, 0))
            total_duration = calculate_union_duration(intervals)
            trial_durations.append(total_duration)
            pbar.set_description(f"稳定性评估（当前均值：{np.mean(trial_durations):.3f}s）")

    # 计算置信区间
    mean_duration, ci_lower, ci_upper, std_duration = calculate_confidence_interval(trial_durations, confidence)
    # 计算变异系数（衡量相对波动，越小越稳定）
    cv = (std_duration / mean_duration) * 100 if mean_duration > 0 else 0.0

    # 整理结果
    stability_result = {
        "n_trials": n_trials,
        "confidence_level": confidence,
        "mean_total_duration": mean_duration,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_duration": std_duration,
        "cv_percent": cv,  # 变异系数（百分比）
        "trial_durations": trial_durations
    }

    # 打印置信区间报告
    print("\n" + "=" * 80)
    print(f"策略稳定性评估报告（{confidence * 100}%置信区间）")
    print("=" * 80)
    print(f"重复实验次数：{n_trials} 次")
    print(f"总遮蔽时长均值：{mean_duration:.3f} s")
    print(f"{confidence * 100}%置信区间：[{ci_lower:.3f} s, {ci_upper:.3f} s]")
    print(f"标准差：{std_duration:.3f} s")
    print(f"变异系数：{cv:.1f}%（<5%为稳定，5%-10%为较稳定，>10%为波动较大）")
    print("\n稳定性结论：")
    if cv < 5:
        print("策略稳定性优秀：多次计算结果波动小，置信区间窄，结果可靠")
    elif cv < 10:
        print("策略稳定性良好：多次计算结果波动适中，置信区间合理，结果可接受")
    else:
        print("策略稳定性一般：多次计算结果波动较大，建议增加实验次数或优化参数")
    print("=" * 80)
    return stability_result


if __name__ == "__main__":
            main()