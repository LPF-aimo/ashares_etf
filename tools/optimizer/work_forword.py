# -*- coding: utf-8 -*-
"""
独立压测脚本 - Walk Forward Analysis
包含蒙特卡洛百次轰炸和全天候四季压测
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
# 添加当前目录到路径，以便导入engine模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接导入engine模块中的BacktestEngine类
from engine import BacktestEngine

def load_config(file_path):
    """加载配置文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    if not hvn_prices:
        return calc_price, False  # 返回是否修改的标志
    
    # 将字符串形式的筹码峰转换为浮点数列表
    try:
        hvn_list = eval(hvn_prices) if isinstance(hvn_prices, str) else hvn_prices
    except:
        return calc_price, False  # 返回是否修改的标志
    
    if not hvn_list or not isinstance(hvn_list, list):
        return calc_price, False  # 返回是否修改的标志
    
    # 磁吸阈值：默认限制在 0.5 个 ATR 范围内
    magnet_threshold = atr * threshold_mult
    
    # 寻找最近的筹码峰
    closest_hvn = min(hvn_list, key=lambda x: abs(x - calc_price))
    
    # 只有在距离足够近时才执行"吸附"
    if abs(closest_hvn - calc_price) <= magnet_threshold:
        # 吸附权重：向筹码峰方向偏移 30% 的差距 (温和吸附，不改变原有阶梯结构)
        adjusted_price = calc_price + (closest_hvn - calc_price) * 0.3
        return adjusted_price, True  # 返回修改后的价格和修改标志
    else:
        return calc_price, False  # 返回原始价格和未修改标志

def calculate_technical_indicators(df):
    """
    计算必要的技术指标 (V8.9 GARCH 预测波幅同步升级版)
    """
    df = df.copy()
    
    # ==========================================
    # 1. V8.9 GARCH/EWMA ATR 预测模型同步
    # ==========================================
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['Raw_ATR'] = true_range.rolling(window=14).mean()
    
    # GARCH/EWMA 核心预测逻辑
    returns = df['Close'].pct_change()
    predicted_volatility_pct = returns.ewm(span=14, adjust=False).std()
    df['GARCH_ATR'] = predicted_volatility_pct * df['Close']
    
    # 平滑混合与物理底线容错
    df['ATR'] = (df['GARCH_ATR'] * 0.7) + (df['Raw_ATR'] * 0.3)
    df['ATR'] = df['ATR'].fillna(df['Raw_ATR']).fillna(0.02 * df['Close'])
    min_atr = df['Close'] * 0.001
    df['ATR'] = df['ATR'].clip(lower=min_atr)
    
    # ==========================================
    # 2. Bias (相对于移动平均的偏差)
    # ==========================================
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Bias'] = (df['Close'] - df['MA10']) / df['MA10']
    
    # ==========================================
    # 3. MACD
    # ==========================================
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD_Line'] = exp1 - exp2
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line']
    
    # ==========================================
    # 4. ADX (基于最新的 GARCH ATR)
    # ==========================================
    up_move = df['High'] - df['High'].shift()
    down_move = df['Low'].shift() - df['Low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / df['ATR'])
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / df['ATR'])
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(window=14).mean()
    
    # ==========================================
    # 5. VAP - 价格成交量分布 (筹码分布算法)
    # ==========================================
    # 计算60日窗口内的筹码密集峰值
    df['VAP_HVN'] = calculate_volume_profile_series(df, window=60, bins=50)
    
    # 填充NaN值
    df = df.fillna(df.bfill()).fillna(df.ffill())
    
    return df

def monte_carlo_bombing(df, config, iterations=100, noise_level=0.002):
    """
    蒙特卡洛百次轰炸测试
    """
    print("="*50)
    print("🎯 蒙特卡洛百次轰炸测试")
    print("="*50)
    
    positive_count = 0
    engine = BacktestEngine()
    
    for i in range(iterations):
        # 每次都复制原始配置
        test_config = config.copy()
        test_config['noise_level'] = noise_level
        # 使用配置文件中的开关设置，而不是硬编码True
        
        result = engine.run(df, test_config)
        net_profit = result['metrics']['net_profit']
        if net_profit > 0:
            positive_count += 1
        
        if (i + 1) % 20 == 0:
            print(f"已完成 {i+1}/{iterations} 次测试...")
    
    win_rate = positive_count / iterations
    print(f"\n📊 测试结果:")
    print(f"   正收益次数: {positive_count}/{iterations}")
    print(f"   净利润胜率: {win_rate:.2%}")
    
    if win_rate > 0.8:
        print("   🏆 评级: 抗扰动极强 (胜率 > 80%)")
    else:
        print(f"   ⚠️  评级: 抗扰动一般 (胜率 <= 80%)")
    
    return win_rate

def all_weather_seasons_test(df, config):
    """
    全天候四季压测 - 修复数据重叠问题，实现基于真实收益率的动态市场状态打标
    """
    print("\n" + "="*50)
    print("🌤️  全天候四季压测")
    print("="*50)
    
    engine = BacktestEngine()
    
    # 计算数据总长度并进行硬切割
    total_days = len(df)
    chunk_size = total_days // 3
    
    # 创建三个不重叠的数据切片
    chunk1 = df.iloc[0:chunk_size].copy()
    chunk2 = df.iloc[chunk_size:2*chunk_size].copy()
    chunk3 = df.iloc[2*chunk_size:].copy()
    
    # 计算每个切片的自然收益率
    def calculate_return(chunk):
        if len(chunk) == 0:
            return 0
        start_price = chunk['Close'].iloc[0]
        end_price = chunk['Close'].iloc[-1]
        return (end_price - start_price) / start_price
    
    chunks_with_returns = [
        ('chunk1', chunk1, calculate_return(chunk1)),
        ('chunk2', chunk2, calculate_return(chunk2)),
        ('chunk3', chunk3, calculate_return(chunk3))
    ]
    
    # 按收益率从低到高排序
    sorted_chunks = sorted(chunks_with_returns, key=lambda x: x[2])
    
    # 根据收益率分配状态标签
    dip_chunk = sorted_chunks[0][1]  # 收益率最低 -> 极寒绞肉区
    choppy_chunk = sorted_chunks[1][1]  # 收益率居中 -> 复杂震荡区
    rise_chunk = sorted_chunks[2][1]  # 收益率最高 -> 相对强势区
    
    results = {}
    
    # 极寒绞肉区测试 (dip)
    print("\n📉 极寒绞肉区测试...")
    if len(dip_chunk) > 1:  # 确保有足够的数据进行回测
        start_date = dip_chunk['Date'].iloc[0].strftime('%Y-%m-%d')
        end_date = dip_chunk['Date'].iloc[-1].strftime('%Y-%m-%d')
        natural_return = calculate_return(dip_chunk)
        
        # 使用配置文件中的开关设置
        test_config = config.copy()
        result_dip = engine.run(dip_chunk, test_config)
        dip_net_profit = result_dip['metrics']['net_profit']
        dip_max_dd = result_dip['metrics']['max_drawdown']
        dip_trade_count = len(result_dip['trade_log'])
        dip_total_return = result_dip['metrics']['total_return']
        dip_initial_cash = config.get('initial_cash', 100000.0)
        dip_final_equity = dip_initial_cash + dip_net_profit
        dip_vap_count = result_dip['metrics'].get('vap_count', 0)  # 获取磁吸介入次数
        
        results['dip'] = {
            'net_profit': dip_net_profit, 
            'max_drawdown': dip_max_dd,
            'trade_count': dip_trade_count,
            'total_return': dip_total_return,
            'final_equity': dip_final_equity,
            'vap_count': dip_vap_count
        }
        
        print(f"   时间区间: {start_date} 至 {end_date}")
        print(f"   ETF自然涨跌幅: {natural_return:.2%}")
        print(f"   策略降本幅度: {dip_net_profit:.2f}")
        print(f"   最大回撤: {dip_max_dd:.2%}")
        print(f"   交易次数: {dip_trade_count}")
        print(f"   磁吸介入次数: {dip_vap_count}")
    else:
        print("   ❌ 数据不足，无法测试")
        results['dip'] = {'net_profit': 0, 'max_drawdown': 0, 'trade_count': 0, 'total_return': 0, 'final_equity': config.get('initial_cash', 100000.0), 'vap_count': 0}
    
    # 复杂震荡区测试 (choppy)
    print("\n🌊 复杂震荡区测试...")
    if len(choppy_chunk) > 1:  # 确保有足够的数据进行回测
        start_date = choppy_chunk['Date'].iloc[0].strftime('%Y-%m-%d')
        end_date = choppy_chunk['Date'].iloc[-1].strftime('%Y-%m-%d')
        natural_return = calculate_return(choppy_chunk)
        
        # 使用配置文件中的开关设置
        test_config = config.copy()
        result_choppy = engine.run(choppy_chunk, test_config)
        choppy_net_profit = result_choppy['metrics']['net_profit']
        choppy_max_dd = result_choppy['metrics']['max_drawdown']
        choppy_trade_count = len(result_choppy['trade_log'])
        choppy_total_return = result_choppy['metrics']['total_return']
        choppy_initial_cash = config.get('initial_cash', 100000.0)
        choppy_final_equity = choppy_initial_cash + choppy_net_profit
        choppy_vap_count = result_choppy['metrics'].get('vap_count', 0)  # 获取磁吸介入次数
        
        results['choppy'] = {
            'net_profit': choppy_net_profit, 
            'max_drawdown': choppy_max_dd,
            'trade_count': choppy_trade_count,
            'total_return': choppy_total_return,
            'final_equity': choppy_final_equity,
            'vap_count': choppy_vap_count
        }
        
        print(f"   时间区间: {start_date} 至 {end_date}")
        print(f"   ETF自然涨跌幅: {natural_return:.2%}")
        print(f"   策略降本幅度: {choppy_net_profit:.2f}")
        print(f"   最大回撤: {choppy_max_dd:.2%}")
        print(f"   交易次数: {choppy_trade_count}")
        print(f"   磁吸介入次数: {choppy_vap_count}")
    else:
        print("   ❌ 数据不足，无法测试")
        results['choppy'] = {'net_profit': 0, 'max_drawdown': 0, 'trade_count': 0, 'total_return': 0, 'final_equity': config.get('initial_cash', 100000.0), 'vap_count': 0}
    
    # 相对强势区测试 (rise)
    print("\n📈 相对强势区测试...")
    if len(rise_chunk) > 1:  # 确保有足够的数据进行回测
        start_date = rise_chunk['Date'].iloc[0].strftime('%Y-%m-%d')
        end_date = rise_chunk['Date'].iloc[-1].strftime('%Y-%m-%d')
        natural_return = calculate_return(rise_chunk)
        
        # 使用配置文件中的开关设置
        test_config = config.copy()
        result_rise = engine.run(rise_chunk, test_config)
        rise_net_profit = result_rise['metrics']['net_profit']
        rise_max_dd = result_rise['metrics']['max_drawdown']
        rise_trade_count = len(result_rise['trade_log'])
        rise_total_return = result_rise['metrics']['total_return']
        rise_initial_cash = config.get('initial_cash', 100000.0)
        rise_final_equity = rise_initial_cash + rise_net_profit
        rise_vap_count = result_rise['metrics'].get('vap_count', 0)  # 获取磁吸介入次数
        
        results['rise'] = {
            'net_profit': rise_net_profit, 
            'max_drawdown': rise_max_dd,
            'trade_count': rise_trade_count,
            'total_return': rise_total_return,
            'final_equity': rise_final_equity,
            'vap_count': rise_vap_count
        }
        
        print(f"   时间区间: {start_date} 至 {end_date}")
        print(f"   ETF自然涨跌幅: {natural_return:.2%}")
        print(f"   策略降本幅度: {rise_net_profit:.2f}")
        print(f"   最大回撤: {rise_max_dd:.2%}")
        print(f"   交易次数: {rise_trade_count}")
        print(f"   磁吸介入次数: {rise_vap_count}")
    else:
        print("   ❌ 数据不足，无法测试")
        results['rise'] = {'net_profit': 0, 'max_drawdown': 0, 'trade_count': 0, 'total_return': 0, 'final_equity': config.get('initial_cash', 100000.0), 'vap_count': 0}
    
    return results


def main():
    """
    主函数 - 重构后的配置适配层
    """
    parser = argparse.ArgumentParser(description='独立压测脚本 - Walk Forward Analysis')
    parser.add_argument('--ticker', type=str, required=True, help='ETF代码')
    args = parser.parse_args()

    print(f"🚀 开始独立压测分析 - 标的: {args.ticker}")
    
    # 设置当前工作目录为脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(parent_dir)
    
    # 检查configs目录是否存在
    configs_dir = os.path.join(parent_dir, 'configs')
    if not os.path.exists(configs_dir):
        print(f"❌ 错误: configs目录不存在: {configs_dir}")
        return
    
    # 检查data目录是否存在
    data_dir = os.path.join(parent_dir, 'data')
    if not os.path.exists(data_dir):
        print(f"❌ 错误: data目录不存在: {data_dir}")
        return
    
    # ==========================================
    # 双数据源加载 (Dynamic Data Loading)
    # ==========================================
    # 1. 加载配置文件
    config_path = os.path.join(configs_dir, f'{args.ticker}_params.json')
    if not os.path.exists(config_path):
        print(f"❌ 错误: 未找到配置文件: {config_path}")
        return
        
    print(f"📁 加载配置文件: {config_path}")
    raw_config = load_config(config_path)
    
    # 2. 加载账户文件
    account_path = os.path.join(configs_dir, f'{args.ticker}_account.json')
    if not os.path.exists(account_path):
        print(f"❌ 错误: 未找到账户文件: {account_path}")
        return
    
    print(f"💼 加载账户文件: {account_path}")
    account_data = load_config(account_path)
    
    # ==========================================
    # 配置适配层 (Adapter Layer) - 动态映射
    # ==========================================
    config = {}
    
    # 1. 动态账户映射 (Cash and Shares Mapping)
    config['initial_cash'] = account_data.get('cash', 10000.0)
    
    # 🚨 致命修复：必须优先读取 params.json (raw_config) 中 AI 寻优出来的最优 base_shares！
    # 只有在 params.json 没有该字段时，才退化去读取 account.json 的物理预设。
    # 🚨 V9.0 架构统一：底仓配置彻底与物理账户解耦，只认 params.json 的战术锁定
    config['base_shares'] = raw_config.get('base_shares', 0)
    config['actual_shares'] = account_data.get('actual_shares', account_data.get('shares', 0))
    
    # 2. 动态战术映射 (Tactical Parameters Mapping)
    tactics = raw_config.get('tactics', {})
    config['profit_protect_pct'] = tactics.get('profit_protect_pct', 1.05)
    config['standard_sell_atr_mult'] = tactics.get('standard_sell_atr_mult', 0.8)
    config['gap_up_atr_threshold'] = tactics.get('gap_up_atr_threshold', 1.0)
    config['gap_down_atr_threshold'] = tactics.get('gap_down_atr_threshold', 1.0)
    config['max_hold_days'] = tactics.get('max_hold_days', 20)
    
    # 🚨 补充遗漏的止盈参数映射
    config['base_grid'] = raw_config.get('base_grid_amount', 100)  # 默认100
    config['trailing_stop_atr_mult'] = tactics.get('trailing_stop_atr_mult', 'inf')
    
    # 3. 嵌套状态打平 (Nested States Flattening)
    # 遍历 params.json 中的 states 字典，将嵌套参数打平为小写前缀键名
    states_dict = raw_config.get('states', {})
    for state_key, state_params in states_dict.items():
        # 将状态键转换为小写前缀（如 'A' -> 'a'）
        state_lower = state_key.lower()
        for param_key, param_value in state_params.items():
            # 创建带小写前缀的参数键（如 'a_buy_mult'）
            flat_key = f"{state_lower}_{param_key}"
            config[flat_key] = param_value
    
    # 4. 其他顶层参数映射
    # 映射其他可能的顶层配置项
    config['grid_interval'] = raw_config.get('grid_interval', 0.02)
    config['atr_period'] = raw_config.get('atr_period', 14)
    config['macd_fast'] = raw_config.get('macd_fast', 12)
    config['macd_slow'] = raw_config.get('macd_slow', 26)
    config['macd_signal'] = raw_config.get('macd_signal', 9)
    config['adx_period'] = raw_config.get('adx_period', 14)
    
    # 从配置文件获取筹码引力开关，默认为False
    config['use_vap_gravity'] = raw_config.get('use_vap_gravity', True)
    print(f"⚙️  筹码引力开关设置: {config['use_vap_gravity']}")
    
    # ==========================================
    # 加载历史数据并执行测试
    # ==========================================
    print("📊 加载历史数据...")
    data_path = os.path.join(data_dir, f'{args.ticker}_daily.csv')
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件不存在: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ 成功加载数据: {len(df)} 行")
    
    # 确保日期列是datetime类型
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 计算技术指标
    print("🔧 计算技术指标...")
    df = calculate_technical_indicators(df)
    
    # 检查是否所有必要指标都有值
    required_columns = ['ATR', 'Bias', 'MACD_Hist', 'ADX', 'VAP_HVN']
    missing_cols = [col for col in required_columns if col not in df.columns or df[col].isna().all()]
    if missing_cols:
        print(f"⚠️ 警告: 以下指标缺失: {missing_cols}")
        return
    
    # 执行蒙特卡洛轰炸测试
    monte_carlo_win_rate = monte_carlo_bombing(df, config)
    
    # 执行全天候四季压测
    weather_results = all_weather_seasons_test(df, config)
    
    print("\n" + "="*50)
    print("📋 压测总结")
    print("="*50)
    print(f"蒙特卡洛胜率: {monte_carlo_win_rate:.2%}")
    if weather_results:
        print(f"主升浪降本: {weather_results['rise']['net_profit']:.2f}, 最大回撤: {weather_results['rise']['max_drawdown']:.2%}, 交易次数: {weather_results['rise']['trade_count']}, 磁吸介入: {weather_results['rise']['vap_count']}次")
        print(f"阴跌绞肉降本: {weather_results['dip']['net_profit']:.2f}, 最大回撤: {weather_results['dip']['max_drawdown']:.2%}, 交易次数: {weather_results['dip']['trade_count']}, 磁吸介入: {weather_results['dip']['vap_count']}次")
        print(f"宽幅震荡降本: {weather_results['choppy']['net_profit']:.2f}, 最大回撤: {weather_results['choppy']['max_drawdown']:.2%}, 交易次数: {weather_results['choppy']['trade_count']}, 磁吸介入: {weather_results['choppy']['vap_count']}次")
        total_vap_count = sum([weather_results[key]['vap_count'] for key in weather_results])
        print(f"> 总计磁吸介入次数: {total_vap_count} 次")


if __name__ == "__main__":
    main()
