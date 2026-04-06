# -*- coding: utf-8 -*-
"""
V8.5 市场状态路由器 (HMM 脑核接入版)
优先听从 AI 潜意识判定，降级兼容硬阈值
"""
import pandas as pd

class MarketRouter:
    def __init__(self, config: dict):
        self.config = config
        self.states = self.config['states']
        self.adx_threshold = self.config['adx_threshold']
    
    def _build_state_response(self, state):
        """构建状态响应字典"""
        return {
            'state': state,
            'buy_mult': self.states[state].get('buy_mult'),
            'sell_mult': self.states[state].get('sell_mult'),
            'amount_mult': self.states[state].get('amount_mult', 1.0)
        }
    
    def determine_state(self, row, today_open=None):
        try:
            # 新增：开盘跳空劫持逻辑
            if today_open is not None:
                gap_down_atr_mult = self.config.get('tactics', {}).get('gap_down_atr_threshold', 1.0)
                gap_down_trigger = row['Close'] - (gap_down_atr_mult * row['ATR'])
                gap_up_atr_mult = self.config.get('tactics', {}).get('gap_up_atr_threshold', 1.0)
                gap_up_trigger = row['Close'] + (gap_up_atr_mult * row['ATR'])

                if today_open <= gap_down_trigger:
                    state = 'B'
                    return self._build_state_response(state)
                elif today_open >= gap_up_trigger:
                    state = 'C'
                    return self._build_state_response(state)

            # 原有逻辑：
            if row['Bias'] <= self.states['B']['bias_threshold']:
                state = 'B'
            # 2. AI 脑核
            elif 'HMM_State' in row and pd.notna(row['HMM_State']):
                state = row['HMM_State']
            elif row['ADX'] < self.adx_threshold:
                state = 'A'
            else:
                if row['MACD_Hist'] > 0:
                    state = 'C'
                elif row['MACD_Hist'] < row.get('MACD_Hist_prev', 0):
                    state = 'D'
                else:
                    state = 'E'
            
            # 提取该状态的所有动态乘数 (严格对齐兵工厂压测引擎)
            return self._build_state_response(state)
        except Exception as e:
            raise ValueError(f"状态判定发生致命错误: {e}")

# 示例使用
if __name__ == "__main__":
    # 示例配置
    config = {
        'states': {
            'A': {'buy_mult': 1.0, 'sell_mult': 1.0, 'amount_mult': 1.0},
            'B': {'buy_mult': 1.0, 'sell_mult': 1.0, 'amount_mult': 1.0, 'bias_threshold': -0.07},
            'C': {'buy_mult': 1.0, 'sell_mult': 1.0, 'amount_mult': 1.0},
            'D': {'buy_mult': 1.0, 'sell_mult': 1.0, 'amount_mult': 1.0},
            'E': {'buy_mult': 1.0, 'sell_mult': 1.0, 'amount_mult': 1.0}
        },
        'adx_threshold': 25,
        'tactics': {
            'gap_down_atr_threshold': 1.0,
            'gap_up_atr_threshold': 1.0
        }
    }
    
    router = MarketRouter(config)
    
    # 示例数据
    row = {
        'Bias': -0.05,
        'ADX': 30,
        'MACD_Hist': 0.1,
        'MACD_Hist_prev': 0.05,
        'Close': 55.0,
        'ATR': 1.0
    }
    
    # 测试正常状态判定
    result_normal = router.determine_state(row)
    print("正常状态判定:", result_normal)
    
    # 测试跳空下跌劫持
    result_gap_down = router.determine_state(row, today_open=52.0)  # 明显低于跳空触发线
    print("跳空下跌劫持:", result_gap_down)
    
    # 测试跳空上涨劫持
    result_gap_up = router.determine_state(row, today_open=58.0)  # 明显高于跳空触发线
    print("跳空上涨劫持:", result_gap_up)