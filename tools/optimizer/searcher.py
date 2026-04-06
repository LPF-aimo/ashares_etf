# -*- coding: utf-8 -*-
"""
高阶网格搜索与并发调度中心 (V9.1 饱和轰炸适配版) - 修复版
"""
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import os
import optuna
import multiprocessing
import joblib
import numpy as np

# 屏蔽 Optuna 的冗余日志
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 导入回测引擎
from tools.optimizer.engine import BacktestEngine

# ==========================================
# 进程级全局变量 (杜绝 DataFrame 的 IPC 复制风暴)
# ==========================================
_GLOBAL_DF = None

def _init_worker(df):
    """初始化 Worker 进程，预装载行情数据"""
    global _GLOBAL_DF
    _GLOBAL_DF = df

def _worker_task(config):
    """
    🌟 核心修复：Worker 任务包装函数
    必须放在最外层，供 GridSearcher 的多进程调用
    """
    global _GLOBAL_DF
    engine = BacktestEngine()
    # 使用已经预装载好的全局数据，避免重复传递几百MB的Dataframe
    result = engine.run(_GLOBAL_DF, config)
    return config.copy(), result['metrics']

class OptunaSearcher:
    """ V9.1 贝叶斯寻优器 """
    def __init__(self, n_trials=1000, max_workers=None, seed=42):
        self.n_trials = n_trials 
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.seed = seed 

    def run_optuna_search(self, df, base_config, param_space):
        results_list = []
        global _GLOBAL_DF
        _GLOBAL_DF = df

        def objective(trial):
            config = base_config.copy()
            for key, values in param_space.items():
                if any(isinstance(v, str) for v in values) or float('inf') in values or -float('inf') in values:
                    config[key] = trial.suggest_categorical(key, values)
                elif any(word in key.lower() for word in ['adx', 'shares', 'days', 'period', 'fast', 'slow']):
                    v_min, v_max = int(min(values)), int(max(values))
                    if 'shares' in key.lower():
                        config[key] = trial.suggest_int(key, v_min, v_max, step=100)
                    else:
                        config[key] = trial.suggest_int(key, v_min, v_max)
                else:
                    v_min, v_max = float(min(values)), float(max(values))
                    config[key] = trial.suggest_float(key, v_min, v_max)
            
            engine = BacktestEngine()
            result = engine.run(_GLOBAL_DF, config)
            metrics = result.get('metrics', result)
            total_return = metrics.get('total_return', 0)
            max_drawdown = metrics.get('max_drawdown', 0.0001)
            total_trades = metrics.get('total_trades', 0)
            
            # 🚨 动态价值观切换器 🚨
            mode = config.get('run_mode', 'normal') # 默认是普通模式
            net_profit = metrics.get('net_profit', 0)
            
            if mode == 'unwind':
                        # ⚔️ 活化绞肉模式：必须高频倒腾，拒绝装死！
                        if total_trades < 150:  # 严厉打击“开局全卖然后装死”的作弊 AI
                            score = -99999.0
                        else:
                            # 1. 直接获取净利润 (因为初始总资产在单次寻优中是常量，直接比绝对值即可)
                            # 这包含了做 T 赚的现金，以及高抛低吸积累的筹码价值
                            net_profit = metrics.get('net_profit', 0)
                            
                            # 2. 对数频率乘数：强迫 AI 成为一台永动机
                            frequency_multiplier = np.log10(total_trades)
                            
                            # 3. 终极评分公式
                            # 如果净利润是负的（说明被大盘 Beta 拖累得太狠），我们用 / 缩小惩罚
                            # 如果净利润是正的，我们用 * 放大奖励
                            if net_profit > 0:
                                score = net_profit * frequency_multiplier
                            else:
                                score = net_profit / frequency_multiplier
            
            trial.set_user_attr('total_return', total_return)
            trial.set_user_attr('max_drawdown', max_drawdown)
            trial.set_user_attr('total_trades', total_trades)
            trial.set_user_attr('net_profit', metrics.get('net_profit', 0))
            trial.set_user_attr('score', score)
            return score

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        with joblib.parallel_backend("loky", n_jobs=self.max_workers):
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                row = base_config.copy()
                row.update(trial.params)
                row.update(trial.user_attrs)
                results_list.append(row)

        df_results = pd.DataFrame(results_list)
        return df_results.sort_values(by='score', ascending=False).reset_index(drop=True) if not df_results.empty else pd.DataFrame()

class GridSearcher:
    """ 网格搜索调度器 """
    def __init__(self, max_workers=None):
        if max_workers is None:
            cpu_cores = os.cpu_count() or 2
            max_workers = max(1, cpu_cores - 1)
        self.max_workers = max_workers
    
    def run_grid_search(self, df, base_config, param_grid):
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        task_configs = []
        for combo in param_combinations:
            task_config = base_config.copy()
            for name, value in zip(param_names, combo): task_config[name] = value
            task_configs.append(task_config)
            
        results = []
        # 🌟 修复：initializer 依赖外部定义的 _init_worker，map 依赖 _worker_task
        with ProcessPoolExecutor(max_workers=self.max_workers, initializer=_init_worker, initargs=(df,)) as executor:
            chunk_size = max(1, len(task_configs) // (self.max_workers * 4))
            try:
                # 🌟 核心修复点：这里的 _worker_task 必须能被 pickle 序列化
                for config, metrics in tqdm(executor.map(_worker_task, task_configs, chunksize=chunk_size), 
                                          total=len(task_configs), desc="算力轰炸中"):
                    total_trades = metrics['total_trades']
                    if total_trades == 0: continue
                    total_return, max_drawdown = metrics['total_return'], metrics['max_drawdown']
                    net_profit = metrics.get('net_profit', 0)
                    mode = config.get('run_mode', 'normal')
                    
                    if mode == 'unwind':
                        # ⚔️ 狂战士解套模式：统一使用对数频率乘数
                        if total_trades < 150:
                            score = -99999.0
                        else:
                            frequency_multiplier = np.log10(total_trades)
                            if net_profit > 0:
                                score = net_profit * frequency_multiplier
                            else:
                                score = net_profit / frequency_multiplier
                    else:
                        # 🛡️ 普通防御模式
                        score = total_return / max_drawdown if max_drawdown > 0 else (float('inf') if total_return > 0 else 0.0)
                        if total_return < 0: score = 0.0
                    
                    result_entry = config.copy()
                    result_entry.update({'total_return': total_return, 'max_drawdown': max_drawdown, 
                                         'total_trades': total_trades, 'score': score})
                    results.append(result_entry)
            except Exception as e:
                print(f"\n⚠️ 并发执行中发生错误: {str(e)}")
        
        return pd.DataFrame(results).sort_values(by='score', ascending=False).reset_index(drop=True) if results else pd.DataFrame()