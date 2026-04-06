# -*- coding: utf-8 -*-
"""
V8.5 机构级双向资金管家 (引擎 100% 对齐版)
彻底消灭硬编码，全盘接管动态 JSON 参数
"""

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

class PortfolioManager:
    def __init__(self, config):
        self.config = config
        self.base_grid_amount = self.config['base_grid_amount']
        self.fee_rate = self.config['fee_rate']
        self.min_fee = self.config['min_fee']
        
        # 提取战术保护底线
        self.tactics = self.config.get('tactics', {})
        self.profit_protect_pct = self.tactics.get('profit_protect_pct', 1.05)
        self.default_base_shares = self.tactics.get('base_shares', 31600)
        
        # 初始化持仓相关属性
        self.base_shares = self.tactics.get('base_shares', 31600)
        self.actual_shares = self.tactics.get('actual_shares', 0)
        self.holding_lots = []  # 机动筹码持仓批次
        
    def sync_maneuverable_lots(self, current_price, entry_date="2024-09-07"):
        """
        活化逻辑：同步机动筹码记录
        如果 actual_shares > base_shares，且当前的 trade_state.json 中没有机动筹码记录，
        则自动创建一个初始批次
        """
        maneuverable_shares = self.actual_shares - self.base_shares
        if maneuverable_shares > 0 and len(self.holding_lots) == 0:
            # 创建初始机动筹码批次
            self.holding_lots.append({
                "shares": maneuverable_shares,
                "entry_date": entry_date,
                "entry_price": current_price
            })
    
    def sync_with_real_account(self, real_actual_shares, real_base_shares, current_price, current_date):
        """
        对账自愈逻辑：同步理论持仓与实际持仓
        """
        # 计算目标机动筹码数量
        target_maneuver = real_actual_shares - real_base_shares
        
        # 计算当前账面机动筹码数量
        current_maneuver = sum(lot['shares'] for lot in self.holding_lots)
        
        # 核心对账逻辑
        if target_maneuver == current_maneuver:
            # 情况 A：相等，无需操作
            pass
        elif target_maneuver > current_maneuver:
            # 情况 B：真实持仓多了 - 说明统帅今天买入了
            difference = target_maneuver - current_maneuver
            self.holding_lots.append({
                "shares": difference,
                "entry_date": current_date,
                "entry_price": current_price
            })
        elif target_maneuver < current_maneuver:
            # 情况 C：真实持仓少了 - 说明统帅今天卖出了
            difference = current_maneuver - target_maneuver
            remaining_to_remove = difference
            
            # 按照 FIFO 原则从头部开始扣减
            while remaining_to_remove > 0 and self.holding_lots:
                oldest_lot = self.holding_lots[0]
                if oldest_lot['shares'] <= remaining_to_remove:
                    # 整批移除
                    removed_shares = self.holding_lots.pop(0)['shares']
                    remaining_to_remove -= removed_shares
                else:
                    # 部分移除
                    oldest_lot['shares'] -= remaining_to_remove
                    remaining_to_remove = 0
    
    def get_maneuverable_shares(self):
        """
        机动筹码定义：返回 holding_lots 列表中所有 shares 的总和
        """
        return sum(lot['shares'] for lot in self.holding_lots)
    
    def make_decision(self, current_state, buy_mult, sell_mult, amount_mult, 
                      today_open, current_atr, holding_shares, current_avg_cost, 
                      highest_price=None, base_shares=None, hvn_prices='[]', yesterday_close=None):
        decision = {'buy_order': None, 'sell_order': None}
        base_shares = base_shares if base_shares is not None else self.default_base_shares
        
        # 获取筹码引力开关
        use_vap_gravity = self.config.get('use_vap_gravity', False)
        
        # 在买卖逻辑开始前，计算当天的波动率缩放因子
        baseline_vol_pct = 0.020 
        current_vol_pct = current_atr / yesterday_close if yesterday_close else 0.02
        vol_ratio = baseline_vol_pct / max(current_vol_pct, 0.005)
        
        if current_state == 'C':
            vol_ratio_clipped = max(1.0, min(vol_ratio, 1.5))
        elif current_state == 'B':
            vol_ratio_clipped = max(0.8, min(vol_ratio, 2.0))
        elif current_state == 'D':
            vol_ratio_clipped = max(0.2, min(vol_ratio, 0.8))
        else:
            vol_ratio_clipped = max(0.5, min(vol_ratio, 1.2))
            
        # 计算经过波动率平价修正后的动态子弹金额
        dynamic_trade_cash = self.base_grid_amount * float(amount_mult) * vol_ratio_clipped
        
        # ==========================================
        # 1. 动态买入挂单逻辑 (对齐引擎)
        # ==========================================
        if str(buy_mult) != 'inf' and buy_mult is not None:
            buy_price = today_open - (float(buy_mult) * current_atr)  # 锚定开盘价
            
            if use_vap_gravity:
                buy_price, _ = apply_volume_magnet(buy_price, hvn_prices, current_atr, threshold_mult=0.5)
            
            # 将原来的 trade_cash 替换为 dynamic_trade_cash
            buy_quantity = int(dynamic_trade_cash / buy_price / 100) * 100
            
            if buy_quantity > 0:
                decision['buy_order'] = {
                    'action': 'Buy',
                    'order_type': 'limit',  # 限价单
                    'quantity': buy_quantity,
                    'price': buy_price
                }
                
        # ==========================================
        # 2. 动态卖出挂单逻辑 (对齐引擎)
        # ==========================================
        # 使用机动筹码数量替代简单的 holding_shares - base_shares
        maneuverable_shares = self.get_maneuverable_shares()
        
        if maneuverable_shares > 0:
            # 💡 特殊战术：C 状态追踪止盈
            if current_state == 'C' and highest_price is not None and highest_price > 0:
                if yesterday_close is not None and maneuverable_shares > 0 and yesterday_close > (current_avg_cost * self.profit_protect_pct):
                    # 利润保护已激活，挂出条件触发单（止损单语义）
                    trailing_stop_price = highest_price - (float(sell_mult) * current_atr)
                    
                    if use_vap_gravity:
                        trailing_stop_price, _ = apply_volume_magnet(trailing_stop_price, hvn_prices, current_atr)
                    
                    decision['sell_order'] = {
                        'action': 'Sell',
                        'order_type': 'stop_loss',  # 明确标记为跌破触发的条件单
                        'quantity': maneuverable_shares,
                        'price': trailing_stop_price
                    }
            
            # 💡 常规战术：A/B/D/E 状态常规卖出 (或 C 状态未触发追踪止盈时)
            elif current_state in ['A', 'B', 'D', 'E'] and str(sell_mult) != 'inf' and sell_mult is not None and decision.get('sell_order') is None:
                sell_price = today_open + (float(sell_mult) * current_atr)  # 锚定开盘价
                
                if use_vap_gravity:
                    sell_price, _ = apply_volume_magnet(sell_price, hvn_prices, current_atr)
                
                # 替换为动态子弹金额
                target_sell_quantity = int(dynamic_trade_cash / sell_price / 100) * 100
                
                # 🛡️ 引擎铁律：绝不卖出底仓 (base_shares)
                max_sellable = maneuverable_shares  # 使用机动筹码数量
                actual_sell_quantity = min(target_sell_quantity, max_sellable)
                
                if actual_sell_quantity > 0:
                    decision['sell_order'] = {
                        'action': 'Sell',
                        'order_type': 'limit',  # 限价单
                        'quantity': actual_sell_quantity,
                        'price': sell_price
                    }
                    
        return decision
    
    def calculate_commission(self, amount):
        return max(amount * self.fee_rate, self.min_fee) 

# 示例使用
if __name__ == "__main__":
    # 示例配置
    config = {
        'base_grid_amount': 2000,
        'fee_rate': 0.000085,
        'min_fee': 1.0,
        'use_vap_gravity': True,  # 开启筹码引力
        'tactics': {
            'profit_protect_pct': 1.05,
            'base_shares': 31600,
            'actual_shares': 35000
        }
    }
    
    # 创建资金管理器实例
    pm = PortfolioManager(config)
    
    # 模拟交易决策
    decision = pm.make_decision(
        current_state='A',
        buy_mult=1.0,
        sell_mult=1.0,
        amount_mult=1.0,
        today_open=55.0,
        current_atr=1.0,
        holding_shares=35000,
        current_avg_cost=52.0,
        hvn_prices='[52.0, 54.5, 58.0]',  # 示例筹码峰数据
        yesterday_close=54.8
    )
    
    print("交易决策:")
    print(f"买入订单: {decision['buy_order']}")
    print(f"卖出订单: {decision['sell_order']}")
    
    # 测试关闭筹码引力
    config_no_gravity = config.copy()
    config_no_gravity['use_vap_gravity'] = False
    pm_no_gravity = PortfolioManager(config_no_gravity)
    
    decision_no_gravity = pm_no_gravity.make_decision(
        current_state='A',
        buy_mult=1.0,
        sell_mult=1.0,
        amount_mult=1.0,
        today_open=55.0,
        current_atr=1.0,
        holding_shares=35000,
        current_avg_cost=52.0,
        hvn_prices='[52.0, 54.5, 58.0]',
        yesterday_close=54.8
    )
    
    print("\n关闭筹码引力后的交易决策:")
    print(f"买入订单: {decision_no_gravity['buy_order']}")
    print(f"卖出订单: {decision_no_gravity['sell_order']}")
    
    # 测试不同状态下的波动率平价效果
    print("\n=== 测试不同市场状态下的波动率平价效果 ===")
    for state in ['A', 'B', 'C', 'D', 'E']:
        decision_test = pm.make_decision(
            current_state=state,
            buy_mult=1.0,
            sell_mult=1.0,
            amount_mult=1.0,
            today_open=55.0,
            current_atr=1.5,  # 较高的ATR模拟高波动
            holding_shares=35000,
            current_avg_cost=52.0,
            hvn_prices='[52.0, 54.5, 58.0]',
            yesterday_close=54.8
        )
        print(f"{state} 状态 -> 买入: {'✓' if decision_test['buy_order'] else '✗'}, 卖出: {'✓' if decision_test['sell_order'] else '✗'}")
