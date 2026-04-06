# -*- coding: utf-8 -*-
"""
V2.0 数据炼金术：硬缝合版 (Index History + Real ETF Truth)
逻辑：
1. 确定边界：以 ETF 起始日 (2025-03-14) 为分界点。
2. 历史缩放：将 2014-2025 的指数数据按首日比例缩放，并注入 14% 噪声模拟误差。
3. 真值注入：2025-03-14 之后的数据全部替换为真实 ETF 的 K 线，严禁拟合。
4. 物理对齐：确保缝合点平滑，且 OHLC 逻辑无误。
"""
import pandas as pd
import numpy as np
import os

def mix_data_hard_stitch(index_path, etf_path, output_path, noise_level=0.14):
    print(f"🚀 开始硬缝合炼金：[{index_path}] (历史) + [{etf_path}] (真值) ...")

    # 1. 加载数据
    df_index = pd.read_csv(index_path, parse_dates=['Date']).sort_values('Date')
    df_etf = pd.read_csv(etf_path, parse_dates=['Date']).sort_values('Date')

    # 2. 确定真值起始点 (2025-03-14)
    etf_start_date = df_etf['Date'].min()
    etf_anchor_price = df_etf['Close'].iloc[0]
    
    # 找到指数在这一天的锚定价格
    index_at_anchor = df_index[df_index['Date'] == etf_start_date]
    if index_at_anchor.empty:
        index_at_anchor = df_index[df_index['Date'] < etf_start_date].iloc[-1:]
    
    index_anchor_price = index_at_anchor['Close'].values[0]
    scale_factor = etf_anchor_price / index_anchor_price
    
    print(f"📍 真值分界点: {etf_start_date.strftime('%Y-%m-%d')}")
    print(f"📏 历史缩放比例: {scale_factor:.8f}")

    # 3. 处理历史部分 (2014-05-23 至 ETF 上市前一天)
    df_history = df_index[df_index['Date'] < etf_start_date].copy()
    price_cols = ['Open', 'High', 'Low', 'Close']
    
    for col in price_cols:
        # 按比例缩放历史指数
        df_history[col] = df_history[col] * scale_factor
        # 仅对模拟的历史数据注入噪声，增加策略鲁棒性
        noise = np.random.normal(0, noise_level / 3, len(df_history))
        df_history[col] = df_history[col] * (1 + noise)

    # 修复历史数据的 OHLC 逻辑
    df_history['High'] = df_history[['Open', 'Close', 'High']].max(axis=1)
    df_history['Low'] = df_history[['Open', 'Close', 'Low']].min(axis=1)
    
    # 历史成交量缩放 (匹配 ETF 现有的平均量级)
    vol_scale = df_etf['Volume'].mean() / df_index['Volume'].mean()
    df_history['Volume'] = df_history['Volume'] * vol_scale
    df_history['Amount'] = df_history['Amount'] * vol_scale * scale_factor

    # 4. 合并数据：历史模拟 + 100% 真实 ETF 数据
    # 🚨 这里直接拼接，不经过任何拟合处理
    df_final = pd.concat([df_history, df_etf], ignore_index=True)
    
    # 5. 最终检查与导出
    df_final = df_final.sort_values('Date').reset_index(drop=True)
    df_final.to_csv(output_path, index=False)
    
    print(f"✅ 炼金完成！")
    print(f"📊 模拟历史样本: {len(df_history)} 天")
    print(f"💎 真实交易样本: {len(df_etf)} 天")
    print(f"📂 最终产出文件: {output_path}")

if __name__ == "__main__":
    # 请确保路径与您的文件名一致
    index_file = "data/197260_train.csv"
    etf_file = "data/159206_daily.csv"
    output_file = "data/159206_mixed_long.csv"
    
    if os.path.exists(index_file) and os.path.exists(etf_file):
        mix_data_hard_stitch(index_file, etf_file, output_file)
    else:
        print("❌ 错误：请检查 data 目录下的源文件是否存在。")