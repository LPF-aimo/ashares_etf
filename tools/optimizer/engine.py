# -*- coding: utf-8 -*-
"""
V7.0 无状态沙盘模拟器 - 热更新版
纯内存回测引擎，严格执行状态机逻辑，杜绝未来函数
修复：C状态卖出滑点错误，卖出不改变均价，重写机动筹码成本公式，废弃胜率计算
新增：真实费率系统、筹码引力开关、磁吸介入统计
"""
import pandas as pd
import numpy as np
from datetime import datetime
import random

def apply_volume_magnet(calc_price, hvn_prices, atr, threshold_mult=0.5):
    """
    筹码引力修正因子
    calc_price: 原本算出的网格价
    hvn_prices: 筹码峰价格列表
    atr: 当前波动率
    threshold_mult: 磁吸半径系数
    """
    if not hvn_prices:
        return calc_price, False
    
    # 将字符串形式的筹码峰转换为浮点数列表
    try:
        hvn_list = eval(hvn_prices) if isinstance(hvn_prices, str) else hvn_prices
    except:
        return calc_price, False
    
    if not hvn_list or not isinstance(hvn_list, list):
        return calc_price, False
    
    # 磁吸阈值：默认限制在 0.5 个 ATR 范围内
    magnet_threshold = atr * threshold_mult
    
    # 寻找最近的筹码峰
    closest_hvn = min(hvn_list, key=lambda x: abs(x - calc_price))
    
    # 只有在距离足够近时才执行"吸附"
    if abs(closest_hvn - calc_price) <= magnet_threshold:
        # 吸附权重：向筹码峰方向偏移 30% 的差距 (温和吸附，不改变原有阶梯结构)
        adjusted_price = calc_price + (closest_hvn - calc_price) * 0.3
        return adjusted_price, True
    else:
        return calc_price, False

class BacktestEngine:
    """
    无状态沙盘模拟器
    仅接收DataFrame和配置字典，返回标准化回测结果
    """
    
    def __init__(self):
        pass
    
    def run(self, df, config):
        """
        执行回测主逻辑 (V9.0 T+1 严格装甲版)
        """
        # 初始化账户状态
        initial_cash = config.get('initial_cash', 100000.0)
        base_shares = config.get('base_shares', 31600)
        actual_shares = config.get('actual_shares', 31600)
        
        # 获取费率配置
        fee_rate = config.get('fee_rate', 0.000085)  # 默认万分之0.85
        min_fee = config.get('min_fee', 1.0)  # 默认最小手续费1元
        
        # 获取噪声级别和引力开关
        noise_level = config.get('noise_level', 0.0)
        use_vap_gravity = config.get('use_vap_gravity', False)
        
        cash = initial_cash
        shares = actual_shares  # 底仓固定不变
        holding_lots = []  # 机动筹码持仓列表
        if actual_shares > base_shares:
            unlocked_shares = actual_shares - base_shares
            holding_lots.append({
                'shares': unlocked_shares,
                'entry_date': df.iloc[0]['Date'].strftime('%Y-%m-%d') if hasattr(df.iloc[0]['Date'], 'strftime') else str(df.iloc[0].get('Date', 'Day_0')),
                'entry_price': df.iloc[0]['Close']
            })
        
        highest_price = 0.0  # C状态下的最高价记录
        equity_curve = []
        trade_log = []
        vap_count = 0
        
        # 遍历历史数据 (从第2天开始)
        for i in range(1, len(df)):
            today = df.iloc[i]
            yesterday = df.iloc[i-1]
            current_date_str = today.get('Date', f'Day_{i}')
            
            # 1. 注入噪声
            if noise_level > 0:
                today = df.iloc[i].copy() 
                noise = noise_level
                today['Low'] = today['Low'] * (1 + random.uniform(-noise, noise))
                today['High'] = today['High'] * (1 + random.uniform(-noise, noise))
                today['Open'] = today['Open'] * (1 + random.uniform(-noise, noise))
            
            # 2. 状态劫持与判定
            ma20_yesterday = yesterday['Close'] / (1 + yesterday['Bias'])
            b_bias_trigger = ma20_yesterday * (1 + config.get('b_bias_threshold', -0.07))
            gap_down_atr_mult = config.get('gap_down_atr_threshold', 1.0)
            gap_down_atr_trigger = yesterday['Close'] - (gap_down_atr_mult * yesterday['ATR'])
            gap_up_atr_mult = config.get('gap_up_atr_threshold', 1.0)
            gap_up_trigger = yesterday['Close'] + (gap_up_atr_mult * yesterday['ATR'])

            if today['Open'] <= b_bias_trigger or today['Open'] <= gap_down_atr_trigger:
                current_state = 'B'
            elif today['Open'] >= gap_up_trigger:
                current_state = 'C'
            else:
                current_state = self._determine_state(yesterday, config)

            # 🛡️【核心修复 1】：T+1 快照！记录今天期初合法可卖的旧筹码总数
            sellable_shares_today = max(0, shares - base_shares)
            
            # 3. 时间衰减检查（只会卖出符合天数的旧筹码）
            max_hold_days = config.get('max_hold_days', 20)
            if current_state != 'B':
                while holding_lots:
                    oldest_lot = holding_lots[0]
                    days_held = 0
                    if isinstance(oldest_lot['entry_date'], str):
                        try:
                            entry_date = datetime.strptime(oldest_lot['entry_date'], '%Y-%m-%d')
                            current_date = datetime.strptime(current_date_str, '%Y-%m-%d')
                            days_held = (current_date - entry_date).days
                        except: pass

                    if days_held > max_hold_days:
                        lot_to_sell = holding_lots.pop(0)
                        sell_shares = lot_to_sell['shares']
                        execution_price = today['Open']
                        trade_value = sell_shares * execution_price
                        fee = max(trade_value * fee_rate, min_fee)
                        cash += (trade_value - fee)
                        shares -= sell_shares
                        sellable_shares_today -= sell_shares  # 同步扣减今日可卖额度

                        trade_log.append({
                            'date': current_date_str,
                            'action': 'Time_Exit_Sell',
                            'price': execution_price,
                            'shares': sell_shares,
                            'cash_left': cash,
                            'fee': fee
                        })
                    else:
                        break 
            
            # 4. 非C状态重置最高价
            if current_state != 'C':
                highest_price = 0.0
            
            # 5. 通用买入逻辑 (买入的新筹码绝对不计入 sellable_shares_today)
            if current_state in ['A', 'B', 'C', 'D', 'E']:
                buy_mult_key = f'{current_state.lower()}_buy_mult'
                if buy_mult_key in config:
                    mult = config[buy_mult_key]
                    if str(mult).lower() == 'inf':
                        buy_price = -float('inf')  
                    else:
                        buy_price = today['Open'] - (float(mult) * yesterday['ATR'])
                        if use_vap_gravity:
                            hvn_prices = yesterday.get('VAP_HVN', '[]')  
                            buy_price, modified = apply_volume_magnet(buy_price, hvn_prices, yesterday['ATR'], threshold_mult=0.5)
                            if modified: vap_count += 1
                        
                    if today['Low'] <= buy_price:
                        execution_price = min(today['Open'], buy_price)
                        amount_mult_key = f'{current_state.lower()}_amount_mult'
                        base_multiplier = float(config.get(amount_mult_key, 1.0))
                        
                        baseline_vol_pct = 0.020 
                        current_vol_pct = yesterday['ATR'] / yesterday['Close']
                        vol_ratio = baseline_vol_pct / max(current_vol_pct, 0.005)
                        
                        if current_state == 'C': vol_ratio_clipped = max(1.0, min(vol_ratio, 1.5))
                        elif current_state == 'B': vol_ratio_clipped = max(0.8, min(vol_ratio, 2.0))
                        elif current_state == 'D': vol_ratio_clipped = max(0.2, min(vol_ratio, 0.8))
                        else: vol_ratio_clipped = max(0.5, min(vol_ratio, 1.2))
                        
                        trade_amount = config['base_grid'] * base_multiplier * vol_ratio_clipped
                        buy_shares = int(trade_amount / execution_price / 100) * 100
                        
                        if buy_shares > 0:
                            trade_value = buy_shares * execution_price
                            fee = max(trade_value * fee_rate, min_fee)
                            required_cash = trade_value + fee
                            if cash >= required_cash:
                                cash -= required_cash
                                shares += buy_shares # 总股数增加，但今天的可卖额度不变！
                                
                                holding_lots.append({
                                    'shares': buy_shares,
                                    'entry_date': current_date_str, # 标记为今天买入
                                    'entry_price': execution_price
                                })
                                trade_log.append({
                                    'date': current_date_str,
                                    'action': 'Buy',
                                    'price': execution_price,
                                    'shares': buy_shares,
                                    'cash_left': cash,
                                    'fee': fee
                                })
            
            # 6. 通用卖出逻辑 (A, B, D, E)
            if current_state in ['A', 'B', 'D', 'E']:
                sell_mult_key = f'{current_state.lower()}_sell_mult'
                if sell_mult_key in config:
                    mult = config[sell_mult_key]
                    if str(mult).lower() == 'inf':
                        sell_price = float('inf')  
                    else:
                        sell_price = today['Open'] + (float(mult) * yesterday['ATR'])
                        if use_vap_gravity:
                            hvn_prices = yesterday.get('VAP_HVN', '[]')  
                            sell_price, modified = apply_volume_magnet(sell_price, hvn_prices, yesterday['ATR'], threshold_mult=0.5)
                            if modified: vap_count += 1
                        
                    if today['High'] >= sell_price:
                        execution_price = max(today['Open'], sell_price)
                        amount_mult_key = f'{current_state.lower()}_amount_mult'
                        base_multiplier = float(config.get(amount_mult_key, 1.0))
                        
                        baseline_vol_pct = 0.020 
                        current_vol_pct = yesterday['ATR'] / yesterday['Close']
                        vol_ratio = baseline_vol_pct / max(current_vol_pct, 0.005)
                        
                        if current_state == 'C': vol_ratio_clipped = max(1.0, min(vol_ratio, 1.5))
                        elif current_state == 'B': vol_ratio_clipped = max(0.8, min(vol_ratio, 2.0))
                        elif current_state == 'D': vol_ratio_clipped = max(0.2, min(vol_ratio, 0.8))
                        else: vol_ratio_clipped = max(0.5, min(vol_ratio, 1.2))
                        
                        trade_amount = config['base_grid'] * base_multiplier * vol_ratio_clipped
                        target_sell_shares = int(trade_amount / execution_price / 100) * 100
                        
                        # 🛡️【核心修复 2】：卖出额度绝不能超过今日期初的旧筹码！
                        actual_sell_shares = min(target_sell_shares, sellable_shares_today)
                        
                        if actual_sell_shares > 0:
                            sellable_shares_today -= actual_sell_shares # 实时扣减
                            remaining_shares_to_sell = actual_sell_shares
                            
                            while remaining_shares_to_sell > 0 and holding_lots:
                                oldest_lot = holding_lots[0]
                                
                                # ⛔ T+1 物理墙：如果碰到今天刚买入的批次，强制停火！
                                if oldest_lot['entry_date'] == current_date_str:
                                    break
                                    
                                if oldest_lot['shares'] <= remaining_shares_to_sell:
                                    shares_to_sell_from_lot = oldest_lot['shares']
                                    holding_lots.pop(0)  
                                else:
                                    shares_to_sell_from_lot = remaining_shares_to_sell
                                    oldest_lot['shares'] -= shares_to_sell_from_lot
                                
                                trade_value = shares_to_sell_from_lot * execution_price
                                fee = max(trade_value * fee_rate, min_fee)
                                proceeds = trade_value - fee
                                cash += proceeds
                                shares -= shares_to_sell_from_lot
                                remaining_shares_to_sell -= shares_to_sell_from_lot
                            
                            trade_log.append({
                                'date': current_date_str,
                                'action': 'Sell',
                                'price': execution_price,
                                'shares': actual_sell_shares - remaining_shares_to_sell, # 实际成交股数
                                'cash_left': cash,
                                'fee': fee
                            })
            
            # 7. C状态的特殊追踪止盈逻辑
            if current_state == 'C' and shares > base_shares:
                if holding_lots:
                    total_cost = sum(lot['shares'] * lot['entry_price'] for lot in holding_lots)
                    total_shares = sum(lot['shares'] for lot in holding_lots)
                    avg_cost = total_cost / total_shares if total_shares > 0 else 0
                else:
                    avg_cost = 0
                
                if yesterday['Close'] > (avg_cost * config.get('profit_protect_pct', 1.05)):  
                    trailing_stop_mult_key = f'{current_state.lower()}_sell_mult'
                    if trailing_stop_mult_key in config:
                        mult = config[trailing_stop_mult_key]
                        if str(mult).lower() == 'inf':
                            trailing_stop_line = -float('inf')  
                        else:
                            trailing_stop_line = highest_price - (float(mult) * yesterday['ATR'])
                            if use_vap_gravity:
                                hvn_prices = yesterday.get('VAP_HVN', '[]')  
                                trailing_stop_line, modified = apply_volume_magnet(trailing_stop_line, hvn_prices, yesterday['ATR'], threshold_mult=0.5)
                                if modified: vap_count += 1
                    else:
                        trailing_stop_line = highest_price - (yesterday['ATR'] * 1.5)
                        
                    if today['Low'] <= trailing_stop_line:
                        execution_price = min(today['Open'], trailing_stop_line)
                        
                        # 🛡️【核心修复 3】：只清算今日期初的合法旧筹码！
                        if sellable_shares_today > 0:
                            actual_sold_shares = 0
                            
                            # FIFO原则：清空旧持仓批次
                            while holding_lots:
                                # ⛔ T+1 物理墙：碰到今天新买入的筹码，绝不平仓！
                                if holding_lots[0]['entry_date'] == current_date_str:
                                    break
                                
                                lot = holding_lots.pop(0)
                                actual_sold_shares += lot['shares']
                                shares -= lot['shares']
                            
                            if actual_sold_shares > 0:
                                sellable_shares_today -= actual_sold_shares
                                trade_value = actual_sold_shares * execution_price
                                fee = max(trade_value * fee_rate, min_fee)
                                proceeds = trade_value - fee
                                cash += proceeds
                                # 注意：禁止强制让 shares = base_shares，因为新筹码必须留存
                                
                                trade_log.append({
                                    'date': current_date_str,
                                    'action': 'Trailing_Stop_Sell',
                                    'price': execution_price,
                                    'shares': actual_sold_shares,
                                    'cash_left': cash,
                                    'fee': fee
                                })
                                highest_price = 0.0
            
            # 8. 计算今日净值
            equity_value = cash + (shares * today['Close'])
            equity_curve.append(equity_value)
            
            # 9. 更新最高价（C状态）
            if current_state == 'C' and shares > base_shares:
                if today['High'] > highest_price:
                    highest_price = today['High']
        
        # 计算回测指标
        metrics = self._calculate_metrics(equity_curve, initial_cash, trade_log, vap_count)
        
        return {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'trade_log': trade_log
        }
    
    def _determine_state(self, yesterday_row, config):
        """
        根据前一天数据判定状态
        """
        # 1. 绝对防线：物理暴跌防线高于一切 AI 预测
        b_bias = config.get('b_bias_threshold', config.get('bias_threshold', -0.08))
        if yesterday_row['Bias'] <= b_bias:
            return 'B'
        
        # 2. 听从 AI 潜意识判定
        if 'HMM_State' in yesterday_row and pd.notna(yesterday_row['HMM_State']):
            return yesterday_row['HMM_State']
        
        # 3. 后续的原有硬规则逻辑保持不变...
        # 优先判断 B状态：Bias <= config['B_bias_threshold'] 且 ADX >= config['adx_threshold']
        if yesterday_row['Bias'] <= config.get('b_bias_threshold', config.get('bias_threshold', 0)) and yesterday_row['ADX'] >= config.get('adx_threshold', 25):
            return 'B'
        
        # A状态 (强趋势向上)：MACD_Hist > 0 且 ADX >= config['adx_threshold']
        if yesterday_row['MACD_Hist'] > 0 and yesterday_row['ADX'] >= config.get('adx_threshold', 25):
            return 'A'
        
        # C状态 (震荡向上)：MACD_Hist > 0 且 ADX < config['adx_threshold']
        if yesterday_row['MACD_Hist'] > 0 and yesterday_row['ADX'] < config.get('adx_threshold', 25):
            return 'C'
        
        # D状态 (强趋势向下)：MACD_Hist < 0 且 ADX >= config['adx_threshold']
        if yesterday_row['MACD_Hist'] < 0 and yesterday_row['ADX'] >= config.get('adx_threshold', 25):
            return 'D'
        
        # E状态 (震荡向下)：MACD_Hist < 0 且 ADX < config['adx_threshold']
        if yesterday_row['MACD_Hist'] < 0 and yesterday_row['ADX'] < config.get('adx_threshold', 25):
            return 'E'
        
        # 默认返回E状态
        return 'E'
    
    def _calculate_metrics(self, equity_curve, initial_cash, trade_log, vap_count):
        """
        计算回测指标
        """
        if len(equity_curve) == 0:
            return {
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'vap_count': 0
            }
        
        # 总收益率
        final_equity = equity_curve[-1]
        net_profit = final_equity - initial_cash
        total_return = (final_equity - initial_cash) / initial_cash
        
        # 最大回撤
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdowns = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0.0
        
        # 交易次数
        total_trades = len(trade_log)
        
        # 废弃错误的胜率计算：直接强行返回 'win_rate': 0.0 即可
        return {
            'total_return': total_return,
            'net_profit': net_profit,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': 0.0,
            'vap_count': vap_count  # 添加磁吸介入次数
        }  

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
        'Volume': np.random.randint(1000000, 5000000, 100),
        'VAP_HVN': ['[52.0, 54.5, 58.0]' for _ in range(100)]  # 示例筹码峰数据
    })
    sample_data.set_index('Date', inplace=True)
    
    # 示例配置
    config = {
        'initial_cash': 100000,
        'base_shares': 31600,
        'actual_shares': 31600,
        'fee_rate': 0.000085,
        'min_fee': 1.0,
        'noise_level': 0.005,
        'use_vap_gravity': True,  # 开启筹码引力
        'b_bias_threshold': -0.07,
        'gap_down_atr_threshold': 1.0,
        'gap_up_atr_threshold': 1.0,
        'adx_threshold': 25,
        'max_hold_days': 20,
        'profit_protect_pct': 1.05,
        'a_buy_mult': 1.0,
        'a_sell_mult': 1.0,
        'a_amount_mult': 1.0,
        'b_buy_mult': 1.0,
        'b_sell_mult': 1.0,
        'b_amount_mult': 1.0,
        'c_buy_mult': 1.0,
        'c_sell_mult': 1.0,
        'c_amount_mult': 1.0,
        'd_buy_mult': 1.0,
        'd_sell_mult': 1.0,
        'd_amount_mult': 1.0,
        'e_buy_mult': 1.0,
        'e_sell_mult': 1.0,
        'e_amount_mult': 1.0,
        'base_grid': 2000
    }
    
    # 运行回测
    engine = BacktestEngine()
    result = engine.run(sample_data, config)
    
    print("回测结果:")
    print(f"总收益率: {result['metrics']['total_return']:.2%}")
    print(f"净收益: {result['metrics']['net_profit']:.2f}")
    print(f"最大回撤: {result['metrics']['max_drawdown']:.2%}")
    print(f"总交易次数: {result['metrics']['total_trades']}")
    print(f"磁吸介入次数: {result['metrics']['vap_count']}")
    print(f"权益曲线最后10日: {result['equity_curve'][-10:]}")
    
    # 测试磁吸修正函数
    print(f"\n磁吸修正函数测试:")
    original_price = 55.0
    hvn_prices = [52.0, 54.5, 58.0]
    atr = 1.0
    corrected_price, modified = apply_volume_magnet(original_price, str(hvn_prices), atr, threshold_mult=0.5)
    print(f"原始价格: {original_price}, 筹码峰: {hvn_prices}, ATR: {atr}")
    print(f"修正后价格: {corrected_price:.4f}, 是否修改: {modified}") 
    
    # 测试关闭筹码引力的情况
    print(f"\n关闭筹码引力测试:")
    config_no_gravity = config.copy()
    config_no_gravity['use_vap_gravity'] = False
    result_no_gravity = engine.run(sample_data, config_no_gravity)
    print(f"关闭引力后总收益率: {result_no_gravity['metrics']['total_return']:.2%}")
    print(f"关闭引力后净收益: {result_no_gravity['metrics']['net_profit']:.2f}")
    print(f"关闭引力后磁吸介入次数: {result_no_gravity['metrics']['vap_count']}")