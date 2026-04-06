# -*- coding: utf-8 -*-
"""
雷达系统 - 芯片ETF量化核心指标计算器
基于纯Pandas实现，零外部依赖
"""
import pandas as pd
import numpy as np
import warnings

# 忽略 pandas 的计算警告，保持控制台纯净
warnings.filterwarnings('ignore')

def calculate_all_indicators(df):
    """
    计算所有核心技术指标 (纯 Pandas 实现，零外部依赖)
    包含: MACD, ATR, ADX, Bias
    """
    df = df.copy()
    
    # ==========================================
    # 1. MACD (趋势动能引擎: 12, 26, 9)
    # ==========================================
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    # 记录昨天的 MACD 柱，用于捕捉 D 状态(阴跌)和 E 状态(转强)的拐点
    df['MACD_Hist_prev'] = df['MACD_Hist'].shift(1)
    
    # ==========================================
    # 2. ATR (GARCH/EWMA 预测波幅引擎)
    # ==========================================
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['Raw_ATR'] = true_range.rolling(window=14, min_periods=1).mean()
    
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
    # 3. ADX (震荡与趋势判定器: 14日)
    # ==========================================
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    
    # 过滤掉非正向波动
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    # 转换为 pandas Series 以便使用 rolling
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    tr14 = true_range.rolling(window=14, min_periods=1).sum()
    plus_di14 = 100 * (plus_dm.rolling(window=14, min_periods=1).sum() / tr14)
    minus_di14 = 100 * (minus_dm.rolling(window=14, min_periods=1).sum() / tr14)
    
    dx = 100 * (np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14))
    df['ADX'] = dx.rolling(window=14, min_periods=1).mean()
    
    # ==========================================
    # 4. Bias (深渊探测器: 20日乖离率)
    # ==========================================
    ma20 = df['Close'].rolling(window=20, min_periods=1).mean()
    df['Bias'] = (df['Close'] - ma20) / ma20
    
    # ==========================================
    # 5. VAP - 价格成交量分布 (筹码分布算法)
    # ==========================================
    # 计算60日窗口内的筹码密集峰值
    df['VAP_HVN'] = calculate_volume_profile_series(df, window=60, bins=50)
    
    # ==========================================
    # 6. 战场清扫 (容错处理) - 修复Lookahead Bias
    # ==========================================
    # 只将无穷大替换为 NaN，然后将所有 NaN 填充为 0（安全做法，防止未来函数）
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

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

class RadarSystem:
    """
    雷达系统主类
    提供统一的指标计算接口
    """
    
    def __init__(self):
        self.indicator_functions = {
            'macd': self.calculate_macd,
            'atr': self.calculate_atr,
            'adx': self.calculate_adx,
            'bias': self.calculate_bias,
            'vap': self.calculate_vap
        }
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        
        result = pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal_line,
            'MACD_Hist': hist,
            'MACD_Hist_prev': hist.shift(1)
        }, index=df.index)
        
        # 只将无穷大替换为 NaN，然后将所有 NaN 填充为 0（安全做法，防止未来函数）
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def calculate_atr(self, df, period=14):
        """计算ATR指标 (V8.9 GARCH/EWMA 预测版)"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        raw_atr = true_range.rolling(window=period, min_periods=1).mean()
        
        # GARCH/EWMA 预测
        returns = df['Close'].pct_change()
        predicted_volatility_pct = returns.ewm(span=period, adjust=False).std()
        garch_atr = predicted_volatility_pct * df['Close']
        
        atr = (garch_atr * 0.7) + (raw_atr * 0.3)
        atr = atr.fillna(raw_atr).fillna(0.02 * df['Close'])
        atr = atr.clip(lower=df['Close'] * 0.001)
        
        # 抛出所需字段
        result = pd.DataFrame({
            'ATR': atr,
            'Raw_ATR': raw_atr,
            'GARCH_ATR': garch_atr
        }, index=df.index)
        
        # 只将无穷大替换为 NaN，然后将所有 NaN 填充为 0（安全做法，防止未来函数）
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def calculate_adx(self, df, period=14):
        """计算ADX指标"""
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        
        # 过滤掉非正向波动
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        
        # 转换为 pandas Series 以便使用 rolling
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        tr14 = true_range.rolling(window=period, min_periods=1).sum()
        plus_di14 = 100 * (plus_dm.rolling(window=period, min_periods=1).sum() / tr14)
        minus_di14 = 100 * (minus_dm.rolling(window=period, min_periods=1).sum() / tr14)
        
        dx = 100 * (np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14))
        adx = dx.rolling(window=period, min_periods=1).mean()
        
        result = pd.DataFrame({'ADX': adx}, index=df.index)
        
        # 只将无穷大替换为 NaN，然后将所有 NaN 填充为 0（安全做法，防止未来函数）
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def calculate_bias(self, df, period=20):
        """计算Bias指标"""
        ma = df['Close'].rolling(window=period, min_periods=1).mean()
        bias = (df['Close'] - ma) / ma
        
        result = pd.DataFrame({'Bias': bias}, index=df.index)
        
        # 只将无穷大替换为 NaN，然后将所有 NaN 填充为 0（安全做法，防止未来函数）
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.fillna(0, inplace=True)
        
        return result
    
    def calculate_vap(self, df, window=60, bins=50):
        """计算VAP筹码分布指标"""
        # 计算每个窗口的筹码密集峰值
        def calculate_window_vap(idx):
            end_idx = idx + 1
            start_idx = max(0, end_idx - window)
            window_df = df.iloc[start_idx:end_idx]
            return calculate_volume_profile(window_df, window=min(window, len(window_df)), bins=bins)
        
        vap_results = []
        for i in range(len(df)):
            hvn_list = calculate_window_vap(i)
            # 将列表转换为字符串存储，因为Series不能存储列表
            vap_results.append(str(hvn_list))
        
        result = pd.DataFrame({'VAP_HVN': pd.Series(vap_results, index=df.index)}, index=df.index)
        
        return result
    
    def calculate_single_indicator(self, df, indicator_name, **kwargs):
        """计算单一指标"""
        if indicator_name.lower() not in self.indicator_functions:
            raise ValueError(f"不支持的指标: {indicator_name}")
        
        func = self.indicator_functions[indicator_name.lower()]
        return func(df, **kwargs)
    
    def batch_calculate(self, df):
        """批量计算所有指标"""
        # 先计算MACD
        macd_data = self.calculate_macd(df)
        df = pd.concat([df, macd_data], axis=1)
        
        # 再计算ATR
        atr_data = self.calculate_atr(df)
        df = pd.concat([df, atr_data], axis=1)
        
        # 再计算ADX
        adx_data = self.calculate_adx(df)
        df = pd.concat([df, adx_data], axis=1)
        
        # 再计算Bias
        bias_data = self.calculate_bias(df)
        df = pd.concat([df, bias_data], axis=1)
        
        # 最后计算VAP
        vap_data = self.calculate_vap(df)
        df = pd.concat([df, vap_data], axis=1)
        
        # 对最终结果进行清理
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        return df

# 示例使用方法
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(50, 60, 100),
        'High': np.random.uniform(55, 65, 100),
        'Low': np.random.uniform(45, 55, 100),
        'Close': np.random.uniform(50, 60, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    sample_data.set_index('Date', inplace=True)
    
    # 使用雷达系统计算指标
    radar = RadarSystem()
    
    # 方法1：使用统一函数计算所有指标
    result1 = calculate_all_indicators(sample_data.copy())
    print("使用统一函数计算的结果:")
    print(result1[['Close', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'ADX', 'Bias', 'VAP_HVN']].head(10))
    
    # 方法2：使用雷达系统类
    result2 = radar.batch_calculate(sample_data.copy())
    print("\n使用雷达系统类计算的结果:")
    print(result2[['Close', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'ADX', 'Bias', 'VAP_HVN']].head(10))
    
    # 方法3：单独计算某个指标
    macd_only = radar.calculate_single_indicator(sample_data.copy(), 'macd')
    print("\n单独计算MACD的结果:")
    print(macd_only.head(10))
    
    # 测试VAP指标
    vap_result = calculate_volume_profile(sample_data, window=60, bins=50)
    print(f"\n测试VAP指标，筹码密集峰值: {vap_result[:5]}")  # 只显示前5个
