#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V8.6 策略全维度压测总控 (Universal Orchestrator)
支持 192 核并发轰炸，具备财务安全预警与参数全维度扫描
二阶段执行逻辑：寻优 → 压测
修复版：解决寻优过程中的常见错误
"""
import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import multiprocessing
import numpy as np
import optuna
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 确保项目根目录在导入路径中
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def calculate_volume_profile(df, window=60, bins=50):
    """
    计算价格成交量分布 (VAP)
    返回最近一个窗口内的筹码密集峰值列表 (High Volume Nodes)
    """
    recent_df = df.tail(window)
    if len(recent_df) < window:
        return []
    
    price_min = recent_df['Low'].min()
    price_max = recent_df['High'].max()
    
    if price_min == price_max:  # 如果价格范围为0，返回空列表
        return []
    
    # 按照价格区间进行成交量分箱
    price_bins = np.linspace(price_min, price_max, bins)
    try:
        volume_dist, _ = np.histogram(recent_df['Close'], bins=price_bins, weights=recent_df['Volume'])
    except:
        return []
    
    # 寻找局部极大值（筹码峰）
    hvn_indices = [i for i in range(1, len(volume_dist)-1) 
                   if volume_dist[i] > volume_dist[i-1] and volume_dist[i] > volume_dist[i+1]]
    
    # 返回对应的价格中值
    hvn_prices = [(price_bins[i] + price_bins[i+1])/2 for i in hvn_indices]
    return sorted(hvn_prices)

def calculate_volume_profile_series(df, window=60, bins=50):
    """
    为DataFrame的每一行计算VAP筹码密集峰值，仅基于当前及历史数据
    """
    def calculate_for_row(row_idx):
        # 只考虑当前行及之前的window数据
        start_idx = max(0, row_idx - window)
        subset_df = df.iloc[start_idx:row_idx]
        
        if len(subset_df) < window // 2:  # 如果数据太少，返回None
            return str([])
        
        price_min = subset_df['Low'].min()
        price_max = subset_df['High'].max()
        
        if price_min == price_max:  # 如果价格范围为0，返回空列表
            return str([])
        
        # 按照价格区间进行成交量分箱
        price_bins = np.linspace(price_min, price_max, bins)
        try:
            volume_dist, _ = np.histogram(subset_df['Close'], bins=price_bins, weights=subset_df['Volume'])
        except:
            return str([])
        
        # 寻找局部极大值（筹码峰）
        hvn_indices = []
        for i in range(1, len(volume_dist)-1):
            if volume_dist[i] > volume_dist[i-1] and volume_dist[i] > volume_dist[i+1]:
                hvn_indices.append(i)
        
        # 返回对应的价格中值
        hvn_prices = [(price_bins[i] + price_bins[i+1])/2 for i in hvn_indices]
        return str(sorted(hvn_prices))
    
    # 对每一行应用计算函数
    results = []
    for i in range(len(df)):
        result = calculate_for_row(i)
        results.append(result)
    
    return pd.Series(results, index=df.index)

def apply_volume_magnet(calc_price, hvn_prices, atr, threshold_mult=0.5):
    """
    筹码引力修正因子
    calc_price: 原本算出的网格价
    hvn_prices: 筹码峰价格列表
    atr: 当前波动率
    threshold_mult: 磁吸半径系数
    """
    if not hvn_prices or pd.isna(calc_price) or pd.isna(atr):
        return calc_price
    
    # 将字符串形式的筹码峰转换为浮点数列表
    try:
        hvn_list = eval(hvn_prices) if isinstance(hvn_prices, str) else hvn_prices
    except:
        return calc_price
    
    if not hvn_list or not isinstance(hvn_list, list):
        return calc_price
    
    # 确保ATR不是NaN或无穷大
    if pd.isna(atr) or np.isinf(atr):
        return calc_price
    
    # 磁吸阈值：默认限制在 0.5 个 ATR 范围内
    magnet_threshold = atr * threshold_mult
    
    # 寻找最近的筹码峰
    closest_hvn = min(hvn_list, key=lambda x: abs(x - calc_price))
    
    # 只有在距离足够近时才执行"吸附"
    if abs(closest_hvn - calc_price) <= magnet_threshold:
        # 吸附权重：向筹码峰方向偏移 30% 的差距 (温和吸附，不改变原有阶梯结构)
        adjusted_price = calc_price + (closest_hvn - calc_price) * 0.3
        # 确保调整后的价格不是NaN
        if pd.isna(adjusted_price):
            return calc_price
        return adjusted_price
    
    return calc_price

def main():
    # 1. 路径自动定位
    data_dir = project_root / "data"
    config_dir = project_root / "configs"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 2. 命令行接口
    parser = argparse.ArgumentParser(description='V8.6 兵工厂终极优化器 - 二阶段执行')
    parser.add_argument('--ticker', type=str, required=True, help='ETF代码')
    parser.add_argument('--mode', type=str, choices=['fast', 'full', 'extreme', 'unwind', 'radar'], default='full')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--history_offset', type=int, default=0, help='切除最近N天数据，模拟过去某时刻的决策环境')
    parser.add_argument('--n_trials', type=int, default=10000, help='寻优尝试次数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，确保结果可复现')
    parser.add_argument('--start_date', type=str, default=None, help='回测起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='回测结束日期 (YYYY-MM-DD)')
    args = parser.parse_args()

    # 3. 资源预加载
    csv_path = data_dir / f"{args.ticker}_daily.csv"
    acc_path = config_dir / f"{args.ticker}_account.json"
    prm_path = config_dir / f"{args.ticker}_params.json"

    # 检查文件完整性
    for f in [csv_path, acc_path, prm_path]:
        if not f.exists():
            print(f"❌ 关键文件缺失: {f.name}")
            sys.exit(1)

    # 4. 解耦适配层 (Adapter Layer)
    with open(acc_path, 'r', encoding='utf-8') as f:
        acc_data = json.load(f)
    with open(prm_path, 'r', encoding='utf-8') as f:
        raw_config = json.load(f)

    tactics = raw_config.get('tactics', {})
    
    # 统一修改为：
    config = {
        'initial_cash': acc_data.get('cash', 100000.0),
        'actual_shares': acc_data.get('actual_shares', acc_data.get('shares', 31600)),
        'base_grid': raw_config.get('base_grid_amount', 4000),
        'fee_rate': raw_config.get('fee_rate', 0.0001),
        'min_fee': raw_config.get('min_fee', 1.0),
        'adx_threshold': raw_config.get('adx_threshold', 20),
        'profit_protect_pct': tactics.get('profit_protect_pct', 1.05),
        'standard_sell_atr_mult': tactics.get('standard_sell_atr_mult', 0.8),
        'deep_buy_firepower': tactics.get('deep_buy_firepower_mult', 3.0),
        'deep_buy_atr_mult': tactics.get('deep_buy_atr_mult', 0.89),
        'trailing_stop_atr_mult': tactics.get('trailing_stop_atr_mult', 0.6),
        'bias_threshold': raw_config.get('states', {}).get('B', {}).get('bias_threshold', -0.06),
        'gap_up_atr_threshold': tactics.get('gap_up_atr_threshold', 1.0),  # 新增早盘跳空防卖飞参数
        'gap_down_atr_threshold': tactics.get('gap_down_atr_threshold', 1.0),
        'max_hold_days': tactics.get('max_hold_days', 20),
        'use_vap_gravity': True,  # 启用筹码引力
        'run_mode': args.mode,
        'base_shares': raw_config.get('base_shares', 0)  # 统一修改为从raw_config读取
    }

    # ==========================================
    # ⚠️ 核心修复 1：自动扁平化注入全状态参数
    # 将 states 字典打平，解决"未能捕获任何信号"的 Bug
    # ==========================================
    states_dict = raw_config.get('states', {})
    for state_key, state_params in states_dict.items():
        state_lower = state_key.lower()
        for k, v in state_params.items():
            config[f"{state_lower}_{k}"] = v

    # 财务安全审计：计算最大所需资金
    max_single_order = config['base_grid'] * config['deep_buy_firepower']
    if max_single_order > config['initial_cash']:
        print(f"⚠️ 财务警告: 单笔最大买入需求({max_single_order}) 超过可用现金({config['initial_cash']})!")
        print(f"💡 建议: 请调低 params.json 中的 base_grid_amount 或增加现金。")

    # 5. 数据灌装 (计算指标)
    from core.indicators import calculate_all_indicators
    df_raw = pd.read_csv(csv_path)
    
    # 计算VAP指标
    df_with_vap = df_raw.copy()
    df_with_vap['VAP_HVN'] = calculate_volume_profile_series(df_with_vap, window=60, bins=50)
    
    # 计算其他技术指标
    df_ready = calculate_all_indicators(df_with_vap)
    if args.start_date:
        df_ready = df_ready[df_ready['Date'] >= args.start_date]
    if args.end_date:
        df_ready = df_ready[df_ready['Date'] <= args.end_date]
    if df_ready.empty:
        print(f"❌ 错误: 在日期范围 {args.start_date} 到 {args.end_date} 内未找到任何数据！")
        sys.exit(1)
    
    print(f"📅 数据时间轴已锁定: {df_ready['Date'].iloc[0]} 至 {df_ready['Date'].iloc[-1]}")
    
    if args.mode in ['unwind', 'radar']:
        print(f"✂️ 启动【{args.mode}模式】: 斩断旧时代数据，保留近 1000 个交易日...")
        # 确保只保留最近的行情，让机器学习当前的阴跌/震荡特征
        history_offset = args.history_offset  # 使用命令行参数
        if history_offset > 0 and len(df_ready) > history_offset:
            df_history = df_ready.iloc[:-history_offset].copy()
            df_train = df_history.tail(1000).reset_index(drop=True)
            print(f"⌛ 时空锚点：已切除最近 {history_offset} 天数据，模拟 {df_ready.iloc[-(history_offset+1)]['Date'] if history_offset > 0 else df_ready.iloc[-1]['Date']} 时的决策环境...")
            df_ready = df_train
        else:
            df_ready = df_ready.tail(1000).reset_index(drop=True)
            print(f"⌛ 时空锚点：使用最近1000天数据进行分析...")
    else:
        # 对于其他模式，应用时空锚点但保留更多历史数据
        if args.history_offset > 0 and len(df_ready) > args.history_offset:
            df_history = df_ready.iloc[:-args.history_offset].copy()
            df_ready = df_history.tail(1000).reset_index(drop=True)
            print(f"⌛ 时空锚点：已切除最近 {args.history_offset} 天数据...")

    # 检查是否有足够的数据进行HMM分析
    if len(df_ready) < 200:
        print(f"⚠️ 数据量不足: 当前数据只有 {len(df_ready)} 天，低于最小要求200天")
        print("   将使用全部可用数据进行分析...")
        df_ready = df_ready.reset_index(drop=True)

    from tools.optimizer.hmm_brain import inject_hmm_states
    df_ready = inject_hmm_states(df_ready)
        

    # 6. 定义全维度搜索矩阵 (Search Matrix)
    # 在 Full/Extreme 模式下，我们榨干 192 核算力，不再考虑时间
    if args.mode == 'fast':
        param_grid = {
            'base_grid': [1800,2000,2500,3000],
            # 测算: -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10
            'bias_threshold': np.around(np.arange(-0.02, -0.11, -0.01), 3).tolist(),
            'deep_buy_atr_mult': np.around(np.arange(0.6, 1.5, 0.1), 2).tolist(),
            'trailing_stop_atr_mult': np.around(np.arange(0.3, 1.0, 0.1), 2).tolist(),
            'deep_buy_firepower': [2.5,], # 压测火力倍数
            'profit_protect_pct': [1.05], # 压测保护点
            'A_buy_mult': [0.9, 1.2, 1.4],
            'D_buy_mult': [-0.1, -0.4, -0.6],
        }
    elif args.mode == 'full':
        # 家用电脑优化版：参数名必须对齐引擎的底层键名 (全是小写前缀)!
        param_grid = {
            'base_grid': [1500.0, 4000.0],
            'b_bias_threshold': np.around(np.arange(-0.02, -0.085, -0.015), 3).tolist(),
            'adx_threshold': [18, 20, 25], 
            'b_buy_mult': [0.7, 0.89],     # 替代原 deep_buy_atr_mult
            'c_sell_mult': [0.9, 1.1, 1.3],    # 替代原 trailing_stop_atr_mult
            'b_amount_mult': [2.0, 2.5, 3.0],  # 替代原 deep_buy_firepower
            'a_buy_mult': [0.7, 0.9, 1.1],
            'd_buy_mult': [-0.1, -0.3],
            'gap_up_atr_threshold': [0.5, 0.8, 1.0, 1.2, 1.5],  # 新增早盘跳空防卖飞参数
        }
    # ... 在 if args.mode == 'full': 逻辑之后加入 ...
    elif args.mode == 'unwind':
# 🚨 V9.2 狂战士精确制导网格 (基于 Top10 基因缩圈)
        
        # 🚨 V9.4 HMM画像定制版 - 解套全域寻优矩阵 
        
        actual_s = config.get('actual_shares', 0)
        
        # 🛡️ 架构师绝对物理防线：
        # 死锁老兵 (不参与反T)：20000 股
        # 弹性活化新兵 (参与高频网格)：8000 ~ 11600 股
        base_shares_grid = [20000, 21000, 22000, 23600]

        param_grid = {
            # 1. 资金底座与防线 (释放火力)
            'base_shares': base_shares_grid,
            'base_grid': [1500.0, 2000.0, 2500.0, 3000.0, 3500.0], 
            # 💡 根据画像(B均值-0.048)，向下探测真正的黄金坑
            'b_bias_threshold': [-0.045, -0.055, -0.065, -0.075, -0.085], 
            'adx_threshold': [20, 24, 27, 30, 35], 

            # 2. 状态战术区 
            
            # ☠️ E状态：日均亏损 0.45% (深买快卖，泥鳅战法)
            'e_amount_mult': [0.4, 0.6, 0.8, 1.0], 
            'e_buy_mult': [0.8, 1.0, 1.2, 1.5],    # 必须防守
            'e_sell_mult': [0.2, 0.3, 0.4, 0.5],   # 赚一点就跑

            # 📈 D状态：日均盈利 0.24% (反T主战场)
            'd_amount_mult': [0.4, 0.6, 0.8, 1.0],            
            'd_buy_mult': [0.2, 0.3, 0.4, 0.5], 
            'd_sell_mult': [-0.20, -0.10, 0.0, 0.10, 0.20], # 水下到冲高全覆盖

            # 💰 B状态：日均盈利 0.31%，超跌反弹 (超级重拳)
            'b_amount_mult': [3.0, 4.0, 5.0, 6.0, 7.0], # 💡 火力全开抄底
            'b_buy_mult': [0.2, 0.3, 0.4],  
            'b_sell_mult': [0.6, 0.8, 1.0, 1.2], 

            # 🚀 C状态：高波主升浪 (日均盈利 1.22%，ATR极高)
            'c_amount_mult': [1.0, 1.2, 1.5],            
            'c_buy_mult': [0.2, 0.3, 0.4],               
            'c_sell_mult': [1.2, 1.5, 1.8, 2.0], 
            
            # 🐢 A状态：低波泥潭 (ATR仅 0.0106)
            'a_amount_mult': [0.5, 0.8, 1.0],
            'a_buy_mult': [0.2, 0.4, 0.6],
            'a_sell_mult': [1.2, 1.5, 1.8, 2.0], # 💡 下调卖点，适应低波动

            # 3. 极值防线
            'gap_up_atr_threshold': [0.5, 0.8, 1.1, 1.5],
            'gap_down_atr_threshold': [0.5, 0.8, 1.1, 1.5],
            
            # 坚守时间物理锁
            'max_hold_days': [9999], 
        }
    elif args.mode == 'radar':  # 添加 'radar' 模式的参数网格
        # 寻找最硬的底座：锁死资金和战术，穷举扫描 ADX 和 Bias
        param_grid = {
            'base_grid': [1500.0],               # 锁死资金
            'b_bias_threshold': np.around(np.arange(-0.03, -0.10, -0.005), 3).tolist(), # 宽幅扫描深渊线 (7个点)
            'adx_threshold': [15, 18, 20, 22, 25, 30],                                 # 宽幅扫描震荡线 (6个点)
            # --- 以下战术乘数全部锁死为中庸的"解套反T"均值 ---
            'b_buy_mult': [0.8],      
            'c_sell_mult': [1.0], 
            'a_sell_mult': [0.8],      
            'b_amount_mult': [2.0],    
            'd_buy_mult': [-0.2],
            'gap_up_atr_threshold': [0.5, 0.8, 1.0, 1.2, 1.5],  # 新增早盘跳空防卖飞参数
        }
    else: # extreme 模式：针对服务器的十万级网格
        param_grid = {
            'base_grid': [2000.0, 2500.0, 3000.0],
            'b_bias_threshold': np.around(np.arange(-0.02, -0.10, -0.01), 3).tolist(),
            'adx_threshold': [15, 20, 25, 30],
            'b_buy_mult': np.around(np.arange(0.4, 1.6, 0.1), 2).tolist(),
            'c_sell_mult': np.around(np.arange(0.2, 1.2, 0.1), 2).tolist(),
            'b_amount_mult': [1.5, 2.0, 3.0, 4.0],
            'a_buy_mult': [0.9, 1.2, 1.4],
            'd_buy_mult': [-0.1, -0.4, -0.6],
            'gap_up_atr_threshold': [0.5, 0.8, 1.0, 1.2, 1.5],  # 新增早盘跳空防卖飞参数
        }

    # 7. 导入搜索器类 (提前导入以供整个流程使用)
    from tools.optimizer.searcher import OptunaSearcher, GridSearcher

    # 第一阶段：寻优 Pass 1 - AI 贝叶斯寻优接管
    if args.mode == 'unwind':
        print(f"\n🎯 第一阶段：AI 贝叶斯寻优启动 (Optuna) - 专攻 {args.mode} 模式...")
        # 贝叶斯寻优 1000 次，效果远超网格穷举 10000 次！
        searcher = OptunaSearcher(
            n_trials=args.n_trials, 
            max_workers=args.workers, 
            seed=args.seed  # ⚠️ 注意：这需要您的 searcher.py 也支持 seed 参数
        )
        df_results = searcher.run_optuna_search(df_ready, config, param_grid)
    else:  # 确保 radar 模式也走 GridSearcher 路由
        searcher = GridSearcher(max_workers=args.workers)
        print(f"\n🎯 第一阶段：传统网格寻优 Pass 1 - 按照 {args.mode} 模式运行...")
        df_results = searcher.run_grid_search(df_ready, config, param_grid)

    if df_results.empty:
        print("❌ 寻优失败: 未能捕获到任何有效交易信号，请放宽 Bias 阈值搜索范围。")
        sys.exit(1)
    
    # 找到最优参数组合
    best_idx = df_results['score'].idxmax()
    best_params = df_results.loc[best_idx].to_dict()
    print(f"🏆 寻优完成 - 最优参数组合 Score: {best_params['score']:.4f}")
    print(f"   最优参数: {best_params}")
    # 🌟🌟🌟 核心新增：将最优参数物理写入 JSON 文件 🌟🌟🌟
    params_save_path = config_dir / f"{args.ticker}_params.json"
    
    # 【修复Bug】：将 Optuna 的扁平化参数，反向重构为引擎可读的嵌套 JSON
    nested_params = {
        "base_shares": int(best_params.get("base_shares", 0)),
        "base_grid_amount": best_params.get("base_grid", 4000.0),
        "fee_rate": best_params.get("fee_rate", 0.000085),
        "min_fee": best_params.get("min_fee", 1.0),
        "adx_threshold": int(best_params.get("adx_threshold", 15)),
        "states": {},
        "tactics": {
            "profit_protect_pct": best_params.get("profit_protect_pct", 1.05),
            "standard_sell_atr_mult": best_params.get("standard_sell_atr_mult", 0.8),
            "gap_up_atr_threshold": best_params.get("gap_up_atr_threshold", 1.0),
            "gap_down_atr_threshold": best_params.get("gap_down_atr_threshold", 1.0),
            "trailing_stop_atr_mult": best_params.get("trailing_stop_atr_mult", 0.6),
            "max_hold_days": int(best_params.get("max_hold_days", 6))
        }
    }

    # 动态将 a_buy_mult 这种扁平键，塞回 states["A"]["buy_mult"] 中
    for state_char in ['A', 'B', 'C', 'D', 'E']:
        state_lower = state_char.lower()
        state_dict = {}
        for sub_key in ['bias_threshold', 'buy_mult', 'sell_mult', 'amount_mult']:
            flat_key = f"{state_lower}_{sub_key}"
            if flat_key in best_params:
                # 捕获无穷大参数 (Optuna 生成的 inf 会被 JSON 兼容序列化为 Infinity)
                val = best_params[flat_key]
                # 特殊兼容处理：有时 float('inf') 会引发实盘序列化报错，转为字符串保护
                if val == float('inf'): val = "inf"
                if val == -float('inf'): val = "-inf"
                state_dict[sub_key] = val
                
        if state_dict:
            nested_params["states"][state_char] = state_dict

    # 物理落盘
    with open(params_save_path, 'w', encoding='utf-8') as f:
        json.dump(nested_params, f, ensure_ascii=False, indent=4)
    
    print(f"✨ 最优战术参数已物理注入: {params_save_path}")
    # 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟
    
    # 8. 第二阶段：压测 Pass 2 - 生存能力压测
    print("\n🧪 第二阶段：压测 Pass 2 - 生存能力压测启动...")
    
    # 【修复1】：统一使用新架构的键名进行敏感度扫描
    center_params = {}
    for key in ['b_bias_threshold', 'b_buy_mult', 'c_sell_mult', 'b_amount_mult', 'gap_up_atr_threshold']:
        if key in best_params:
            center_params[key] = best_params[key]
    
    # 创建敏感度参数网格 - 修复敏感度阵列崩溃
    sensitivity_grid = {}
    for key, center_val in center_params.items():
        if isinstance(center_val, (int, float)):
            if center_val == float('inf') or center_val == -float('inf'):
                sensitivity_grid[key] = [center_val] # 遇到无穷大，跳过该参数的浮动扫描
                continue
                
            lower = center_val * 0.95
            upper = center_val * 1.05
            if key in ['b_bias_threshold', 'b_buy_mult', 'c_sell_mult', 'gap_up_atr_threshold']:
                sensitivity_grid[key] = np.linspace(lower, upper, 5).tolist()
            else:
                sensitivity_grid[key] = np.linspace(lower, upper, 3).tolist()
    
    print("🔍 参数敏感度扫描开始...")
    # 【致命 Bug 修复2】：必须使用第一名 best_params 作为底座，绝不能用 base_config！
    # 【修复3】：对于敏感度测试和压力测试，始终使用 GridSearcher
    
    # 修复：将最优参数与基础配置合并，确保所有必要参数都存在
    full_best_config = config.copy()
    full_best_config.update(best_params)
    
    grid_searcher = GridSearcher(max_workers=args.workers)
    sensitivity_results = grid_searcher.run_grid_search(df_ready, full_best_config, sensitivity_grid)
    
    if not sensitivity_results.empty:
        print(f"📊 敏感度分析完成，共测试 {len(sensitivity_results)} 种变体")
        best_sensitivity_score = sensitivity_results['score'].max()
        print(f"   最佳敏感度得分: {best_sensitivity_score:.4f}")
        print(f"   相对于基准下降: {(best_params['score'] - best_sensitivity_score) / best_params['score'] * 100:.2f}%")
    else:
        print("⚠️  敏感度分析未产生有效结果")
    
    # 财务极限压力测试：强行将 initial_cash 减半
    print("\n💸 财务极限压力测试开始...")
    # 【致命 Bug 修复3】：同样必须使用完整的最佳配置
    stress_config = full_best_config.copy()
    stress_config['initial_cash'] = full_best_config['initial_cash'] / 2
    print(f"   压力测试现金: {stress_config['initial_cash']:.2f} (原现金: {full_best_config['initial_cash']:.2f})")
    
    stress_results = grid_searcher.run_grid_search(df_ready, stress_config, {key: [center_params[key]] for key in center_params.keys()})
    
    if not stress_results.empty:
        stress_score = stress_results['score'].iloc[0]
        print(f"   压力测试得分: {stress_score:.4f}")
        print(f"   财务压力损失: {(best_params['score'] - stress_score) / best_params['score'] * 100:.2f}%")
        
        # 检查是否触发逻辑死锁或异常
        if stress_results['total_trades'].iloc[0] == 0:
            print("⚠️  警告: 财务压力下策略触发逻辑死锁（无交易发生）")
        else:
            print("✅ 财务压力测试通过")
    else:
        print("❌ 财务压力测试失败: 策略在减半资金下无法正常运行")
    
    # 9. 生成综合报告 (V9.1 自动归档版)
    from tools.optimizer.auditor import StrategyAuditor
    from datetime import datetime
    
    print("\n📝 正在生成最终审计报告并进行自动归档...")
    auditor = StrategyAuditor()
    
    # 合并所有结果
    combined_results = {
        'primary_optimization': df_results,
        'sensitivity_analysis': sensitivity_results,
        'financial_stress_test': stress_results,
        'best_params': best_params
    }
    
    # 1. 生成报告内容 (Markdown 文本)
    report_md = auditor.generate_report(args.ticker, combined_results, df_ready, config)
    
    # 2. 🌟 核心修改：构建带指纹的文件名
    # 包含：代码、运行模式、随机种子、时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 如果是 Bash 脚本运行，建议文件名包含种子信息，方便横向对比
    seed_str = f"_seed{args.seed}" if hasattr(args, 'seed') else ""
    report_filename = f"audit_{args.ticker}_{args.mode}{seed_str}_{timestamp}.md"
    
    # 3. 确保路径并写入
    report_path = reports_dir / report_filename
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    # 同时保留一个最新副本链接（可选，方便快速查看最新结果）
    latest_link = reports_dir / f"audit_report_{args.ticker}_LATEST.md"
    with open(latest_link, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n✅ 综合审计战报已生成: {report_path}")
    print(f"🔗 最新副本已更新: {latest_link}")


if __name__ == "__main__":
    main()
