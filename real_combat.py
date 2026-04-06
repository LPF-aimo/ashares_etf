# -*- coding: utf-8 -*-
"""
V2.5 参数化重构版每日实盘执行脚本
每天收盘后运行，生成次日挂单指令
重构内容：完全参数化账户状态，移除硬编码，增强健壮性
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime
# 引入 core 核心组件
from core.indicators import calculate_all_indicators
from core.router import MarketRouter
from core.portfolio import PortfolioManager

class TradeStateManager:
    """交易状态管理器 - 本地状态记忆功能"""
    
    def __init__(self, state_file='data/trade_state.json'):
        self.state_file = state_file
        self.state = self.load_state()
    
    def load_state(self):
        """从文件加载状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  读取状态文件失败: {e}")
                return {"highest_price_since_c": 0.0, "holding_lots": []}
        else:
            # 文件不存在，创建默认状态
            # 确保data目录存在
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            return {"highest_price_since_c": 0.0, "holding_lots": []}
    
    def save_state(self):
        """保存状态到文件"""
        try:
            # 确保data目录存在
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"⚠️  保存状态文件失败: {e}")
    
    def update_highest_price(self, current_high):
        """更新C状态下的最高价记录"""
        if current_high > self.state['highest_price_since_c']:
            self.state['highest_price_since_c'] = current_high
            self.save_state()
            return True
        return False
    
    def reset_highest_price(self):
        """重置最高价记录"""
        self.state['highest_price_since_c'] = 0.0
        self.save_state()
    
    def get_highest_price(self):
        """获取当前记录的最高价"""
        return self.state['highest_price_since_c']
    
    def get_holding_lots(self):
        """获取持仓批次列表"""
        return self.state.get('holding_lots', [])
    
    def set_holding_lots(self, lots):
        """设置持仓批次列表"""
        self.state['holding_lots'] = lots
        self.save_state()

def generate_simple_order_card(order_dict, yesterday_close):
    """
    生成简洁的订单信息
    :param order_dict: 订单字典，包含action, quantity, price, order_type等
    :param yesterday_close: 昨日收盘价
    :return: 简洁的订单信息字符串
    """
    if order_dict is None:
        return ""
    
    action = order_dict['action']
    price = order_dict['price']
    quantity = order_dict['quantity']
    order_type = order_dict.get('order_type', 'limit')
    
    # 判断订单类型
    if action == 'Buy':
        if price < yesterday_close:
            order_desc = "【普通限价单 - 低吸】"
            trigger_desc = "立即执行"
        else:
            order_desc = "【条件单/触价单 - 突破追涨】"
            trigger_desc = f"触发条件: 股价 ≥ {price:.4f}"
    else:  # Sell
        if price > yesterday_close:
            order_desc = "【普通限价单 - 高抛】"
            trigger_desc = "立即执行"
        else:
            # 检查是否是条件止损单
            if order_type == 'stop_loss':
                order_desc = "🚨【条件止损单/触价单 - 破位止损】🚨"
                trigger_desc = f"⚠️  这是一个【条件止损单/触价单】！请使用券商的条件单功能，设置【跌破 ¥{price:.4f} 触发卖出】，切勿挂成普通限价单！"
            else:
                order_desc = "【条件单/触价单 - 破位止损】"
                trigger_desc = f"触发条件: 股价 ≤ {price:.4f}"
    
    # 计算金额和差价
    amount = price * quantity
    diff = price - yesterday_close
    diff_percent = (diff / yesterday_close) * 100
    
    # 创建简洁订单信息
    card = f"""  {order_desc}
  价格: ¥{price:.4f} | 数量: {quantity:,}股 | 金额: ¥{amount:.2f}
  {trigger_desc}
  相对昨收: {'+' if diff >= 0 else ''}{diff:.4f} ({'+' if diff_percent >= 0 else ''}{diff_percent:.2f}%)
"""
    return card

def load_account_config(ticker, cash_override=None, base_shares_override=None, actual_shares_override=None):
    """
    加载账户配置文件
    :param ticker: ETF代码
    :param cash_override: 现金覆盖参数
    :param base_shares_override: 底仓覆盖参数
    :param actual_shares_override: 实际持仓覆盖参数
    :return: 账户配置字典
    """
    account_file = f'configs/{ticker}_account.json'
    
    if not os.path.exists(account_file):
        raise FileNotFoundError(f"账户配置文件不存在: {account_file}. 请先创建该文件.")
    
    with open(account_file, 'r', encoding='utf-8') as f:
        account_data = json.load(f)
        # 🚨 升级为 V8.6 的活化字段校验
        required_keys = ['cash', 'actual_shares'] 
        missing_keys = [k for k in required_keys if k not in account_data]
        if missing_keys:
            print(f"❌ 军需库(账户配置)加载失败: 账户配置文件中缺少必要字段: {missing_keys}")
            sys.exit(1)
    
    # 应用命令行覆盖参数
    if cash_override is not None:
        account_data['cash'] = cash_override
    if actual_shares_override is not None:
        account_data['actual_shares'] = actual_shares_override
    
    return account_data

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='V8.7 每日实盘火控雷达')
    parser.add_argument('--ticker', type=str, required=True, help='ETF代码')
    parser.add_argument('--cash', type=float, help='覆盖账户现金余额')
    parser.add_argument('--base-shares', type=int, help='覆盖账户底仓数量')
    parser.add_argument('--actual-shares', type=int, help='覆盖账户实际持仓数量')
    parser.add_argument('--open', type=float, default=None, help='(可选) 今日 9:25 集合竞价开盘价')
    args = parser.parse_args()
    
    ticker = args.ticker
    
    print("\n" + "═"*65)
    print(f" 🦅 【量化兵工厂 V8.5 破壁者】 每日实盘指挥中枢 - [{ticker}]")
    print(f" ⏱️  启动时刻: 机器算力镇压主观情绪，纪律高于一切。")
    print("═"*65)
    
    # 加载账户配置
    try:
        account_config = load_account_config(
            ticker, 
            cash_override=args.cash,
            actual_shares_override=args.actual_shares
        )
        real_cash = account_config['cash']
        real_actual_shares = account_config['actual_shares']
    except Exception as e:
        print(f"❌ 军需库(账户配置)加载失败: {e}")
        sys.exit(1)
    
    # 加载参数配置
    config_path = os.path.join('configs', f'{ticker}_params.json')
    
    if not os.path.exists(config_path):
        print(f"❌ 战术指令(配置文件)丢失: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查必要字段
        required_config_fields = ['tactics', 'base_grid_amount', 'fee_rate', 'adx_threshold', 'states']
        missing_config_fields = [field for field in required_config_fields if field not in config]
        if missing_config_fields:
            raise ValueError(f"战术指令残缺，缺少必要字段: {missing_config_fields}")
            
        if 'tactics' not in config:
            raise ValueError("战术指令残缺，缺少 'tactics' 护盾字段")
            
    except Exception as e:
        print(f"❌ 加载战术指令失败: {e}")
        sys.exit(1)
    
    # 🚨 致命修复：必须优先读取 params.json (config) 中 AI 寻优出来的最优 base_shares！
    # 只有在 params.json 没有该字段时，才退化去读取 account.json 的物理预设。
    optimal_base_shares = config.get('base_shares', None)
    if optimal_base_shares is not None:
        real_base_shares = optimal_base_shares
    else:
        real_base_shares = 0  # 如果params.json中没有base_shares，则默认为0
    
    # 应用命令行覆盖参数
    if args.base_shares is not None:
        real_base_shares = args.base_shares
    
    # 打印军情总览
    print(f"💰 当前可用资金: ¥{real_cash:,.2f}")
    print(f"📦 真实总持仓: {real_actual_shares:,} 股")
    print(f"🛡️ 铁血死锁底仓: {real_base_shares:,} 股")
    print(f"⚔️ 当前活化机动筹码: {real_actual_shares - real_base_shares:,} 股")
    
    # 加载数据
    data_path = os.path.join('data', f'{ticker}_daily.csv')
    if not os.path.exists(data_path):
        print(f"❌ 前线战报(数据文件)丢失: {data_path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"❌ 读取前线战报失败: {e}")
        sys.exit(1)
    
    # 计算指标
    df = calculate_all_indicators(df)
    try:
        from tools.optimizer.hmm_brain import inject_hmm_states
        print("\n🧠 [AI 脑核接入] 正在启动隐马尔可夫高维雷达，剥离市场噪音...")
        # 截取近 250 天数据供 HMM 学习当前情绪
        df_hmm = df.tail(min(len(df), 1000)).reset_index(drop=True) # 增加长度防呆
        df_hmm = inject_hmm_states(df_hmm, train_mode=False) # 🚨 核心：必须为 False
        hmm_result = df_hmm.iloc[-1]['HMM_State']
        print(f"    [架构师 Debug] HMM 底层输出的原始状态标识为: {hmm_result}")
        # 将最新一天的 HMM 状态提取出来，打在总数据上
        df.loc[df.index[-1], 'HMM_State'] = hmm_result
        print("✅ [潜意识注入] HMM 聚类完成，市场底牌已看透！")
    except Exception as e:
        print(f"\n⚠️ HMM 大脑受干扰异常: {e}")
        print("🛡️ [系统降级] 已自动切换至传统 ADX/Bias 物理装甲防御！")
    
    # 初始化状态管理器
    state_file = f'data/{ticker}_trade_state.json'
    state_manager = TradeStateManager(state_file=state_file)
    current_highest_price = state_manager.get_highest_price()
    
    # 2. 实例化组件
    try:
        router = MarketRouter(config)
        portfolio_manager = PortfolioManager(config)
    except Exception as e:
        print(f"❌ 武器系统初始化崩溃: {e}")
        sys.exit(1)
    
    # 3. 提取今天的数据
    today_data = df.iloc[-1]
    
    # 🚨 【V9.2 时空对齐】：早上 9:25 运行时，CSV 最后一行就是"昨天"的数据
    latest_data = df.iloc[-1]      # 这就是我们今天决策的全部基石 (昨收)
    yesterday_close = latest_data['Close']
    latest_atr = latest_data['ATR']
    
    # 🚨 动态开盘雷达接入 & 涨跌停物理拦截锁 (V9.3 强制交互版)
    limit_up = yesterday_close * 1.10   # 涨停极值
    limit_down = yesterday_close * 0.90 # 跌停极值

    if args.open is not None:
        today_open = args.open
        # 命令行模式如果给错参数，直接阻断程序，防止自动化脚本酿成大错
        if today_open > limit_up or today_open < limit_down:
            print(f" ❌ 【致命拦截】命令行传入的开盘价(¥{today_open:.3f})已突破 A股 10% 涨跌停极限(¥{limit_down:.3f} - ¥{limit_up:.3f})！")
            sys.exit(1)
    else:
        print("\n" + "▓"*65)
        while True:
            today_open_str = input(f" 🎯 请输入 [{args.ticker}] 今日 9:25 的开盘价 (直接回车则默认昨收 ¥{yesterday_close:.3f}): ")
            
            try:
                today_open = float(today_open_str) if today_open_str.strip() else yesterday_close
            except ValueError:
                print(" ⚠️ 【格式错误】请输入有效的数字！")
                continue

            # 🛡️ 胖手指防御：拦截离谱数据并要求统帅重新输入
            if today_open > limit_up:
                print(f" ⚠️ 【系统拦截】您输入的开盘价 (¥{today_open:.3f}) 已突破涨停极限 (¥{limit_up:.3f})！")
                print(" 🚫 请检查是否多按了小数点或数字，并重新输入。")
                continue
            elif today_open < limit_down:
                print(f" ⚠️ 【系统拦截】您输入的开盘价 (¥{today_open:.3f}) 已击穿跌停极限 (¥{limit_down:.3f})！")
                print(" 🚫 请检查是否少按了小数点或数字，并重新输入。")
                continue
            else:
                break  # 数据合法，解除拦截，跳出循环

    print("▓"*65 + "\n")
    
    # 执行对账自愈逻辑
    print("🔄 【自愈同步机制启动】正在进行持仓对账...")
    holding_lots = state_manager.get_holding_lots()
    portfolio_manager.holding_lots = holding_lots  # 将当前持仓同步到资金管家
    
    # 获取今天的日期
    current_date = today_data.get('Date', datetime.now().strftime('%Y-%m-%d'))
    
    # 调用自愈同步方法
    portfolio_manager.sync_with_real_account(
        real_actual_shares=real_actual_shares,
        real_base_shares=real_base_shares,
        current_price=today_data['Close'],
        current_date=current_date
    )
    
    # 将修正后的持仓保存回状态管理器
    state_manager.set_holding_lots(portfolio_manager.holding_lots)
    print("✅ 【自愈同步完成】持仓数据已与实际账户对齐")
    
    # 通过 HMM 判定初始状态
    # 通过 HMM 判定初始状态
    try:
        # 🌟 修复：必须传入 latest_data (昨收数据)，而不是错位的 iloc[-2]
        state_info = router.determine_state(latest_data, today_open=today_open)
        current_state = state_info['state']
    except Exception as e:
        print(f"❌ 战场态势研判失败: {e}")
        sys.exit(1)
    
    # 定义各状态的绝对心法 (V9.1 Seed 42 对齐版)
    STATE_MANTRAS = {
        'A': "温和下跌 📉 | 心法：高位防守，AI建议格局死拿 (卖点乘数高达2.0)。",
        'B': "深渊黄金坑 🩸 | 心法：极度恐慌即为战机！执行 1.8x 倍重拳重仓抄底！",
        'C': "疯牛主升浪 🚀 | 心法：主升浪绝不恋战，有利润适度收割止盈。",
        'D': "温和反弹 📈 | 心法：盈亏比极差，近乎关停火力 (0.018x)，严格观望。",
        'E': "极寒放血 🧊 | 心法：持续阴跌区，大幅缩减头寸，猥琐发育摊薄成本。"
    }
    # 更新或重置最高价记录
    if current_state == 'C':
        today_high = today_data['High']
        updated = state_manager.update_highest_price(today_high)
        current_highest_price = state_manager.get_highest_price()
    else:
        state_manager.reset_highest_price()
        current_highest_price = 0.0
    
    # 引入时间衰减检查
    holding_lots = state_manager.get_holding_lots()
    time_decay_orders = []
    current_date_str = today_data.get('Date', f'Day_{len(df)-1}')
    max_hold_days = config['tactics'].get('max_hold_days', 20)
    
    # 遍历 holding_lots，检查是否超期
    if current_state != 'B':  # 只有在非 B 状态时才执行时间衰减
        while holding_lots:
            oldest_lot = holding_lots[0]
            # 安全解析日期
            days_held = 0
            if isinstance(oldest_lot['entry_date'], str):
                try:
                    entry_date = datetime.strptime(oldest_lot['entry_date'], '%Y-%m-%d')
                    current_date_obj = datetime.strptime(current_date_str, '%Y-%m-%d')
                    days_held = (current_date_obj - entry_date).days
                except:
                    pass
            if days_held > max_hold_days:
                # 超期，生成时间衰减卖出指令
                lot_to_sell = holding_lots.pop(0)
                sell_shares = lot_to_sell['shares']
                execution_price = today_open  # 以开盘价成交
                # 如果没有输入开盘价，则使用收盘价
                if args.open is None:
                    execution_price = today_data['Open']  # 实际开盘价
                
                time_decay_order = {
                    'action': 'Sell',
                    'order_type': 'time_decay',
                    'quantity': sell_shares,
                    'price': execution_price
                }
                
                time_decay_orders.append(time_decay_order)
            else:
                break  # 最老的都没过期，后面的肯定没过期，跳出
    
    # 更新持仓批次
    state_manager.set_holding_lots(holding_lots)
    
    # ==========================================
    # V8.7 动态雷达：状态跃迁与最终指令生成
    # ==========================================
    # 1. 计算触发防线
    # 🌟 修复：统一使用对齐后的 latest_data
    ma10_yesterday = yesterday_close / (1 + latest_data['Bias'])
    b_bias_trigger = ma10_yesterday * (1 + config['states']['B']['bias_threshold'])
    gap_down_atr_trigger = yesterday_close - (config['tactics'].get('gap_down_atr_threshold', 1.0) * latest_atr)
    gap_up_trigger = yesterday_close + (config['tactics'].get('gap_up_atr_threshold', 1.0) * latest_atr)
    # 2. 根据 9:25 开盘价，确定今日【最终状态】
    final_state = current_state  # 默认延续昨夜计算的基础状态
    state_reason = "延续昨日趋势，按部就班"
    if today_open <= b_bias_trigger or today_open <= gap_down_atr_trigger:
        final_state = 'B'
        if current_state != 'B':
            state_reason = f"🚨 开盘暴跌触发防线 (开盘价 <= {max(b_bias_trigger, gap_down_atr_trigger):.4f}) -> 强制跃迁至 B 状态！"
    elif today_open >= gap_up_trigger:
        final_state = 'C'
        if current_state != 'C':
            state_reason = f"🚀 开盘暴涨触发防线 (开盘价 >= {gap_up_trigger:.4f}) -> 强制跃迁至 C 状态！"
    # ==========================================
    # 🌟 V8.9 核心：非对称波动率平价 (Asymmetric Volatility Parity)
    # ==========================================
    # 1. 基础参数获取
    base_grid_amount = float(config.get('base_grid_amount', 2000.0))
    amount_mult = float(config['states'][final_state]['amount_mult'])
    
    # 2. 风险系数计算 (基于 GARCH ATR)
    current_vol_pct = latest_atr / today_open
    baseline_vol_pct = 0.020  # ETF 波动率中枢 2%
    vol_ratio = baseline_vol_pct / max(current_vol_pct, 0.005)
    
    # 3. 统帅意志：非对称限幅 (对齐引擎逻辑)
    if final_state == 'C':
        vol_ratio_clipped = max(1.0, min(vol_ratio, 1.5)) # 主升浪：不缩减，可加码
    elif final_state == 'B':
        vol_ratio_clipped = max(0.8, min(vol_ratio, 2.0)) # 黄金坑：允许重拳抄底
    elif final_state == 'D':
        vol_ratio_clipped = max(0.2, min(vol_ratio, 0.8)) # 温和反弹：盈亏比差，强行缩仓
    else:
        vol_ratio_clipped = max(0.5, min(vol_ratio, 1.2)) # A/E 区：标准风控
        
    # 4. 计算最终单笔投入资金
    final_risk_mult = amount_mult * vol_ratio_clipped
    trade_cash = base_grid_amount * final_risk_mult
    
    # 计算当前平均成本
    current_avg_cost = 0.0
    if portfolio_manager.holding_lots:
        total_cost = sum(lot['shares'] * lot['entry_price'] for lot in portfolio_manager.holding_lots)
        total_shares = sum(lot['shares'] for lot in portfolio_manager.holding_lots)
        if total_shares > 0:
            current_avg_cost = total_cost / total_shares
    
    # ==========================================
    # 🌟 核心修正：重新校准决策弹药 (确保状态与参数同步)
    # ==========================================
    
    # 1. 获取最终裁决状态对应的战术参数
    final_tactics = config['states'][final_state]
    
    # 2. 重新应用统帅的『非对称火力强度』(这里必须包含您刚才算的风险溢价)
    # 使用刚才计算出的 final_risk_mult，确保 B 状态能打出 4 倍以上的重拳
    decision_amount_mult = final_risk_mult 

    # 3. 决定买卖间距 (使用最终状态的 mult)
    buy_mult = float(final_tactics.get('buy_mult', 0.8))
    sell_mult = float(final_tactics.get('sell_mult', 0.8))

    # 4. 召唤决策引擎
    decision = portfolio_manager.make_decision(
        current_state=final_state,         # 👈 必须传 final_state (B/C/E...)
        buy_mult=buy_mult,
        sell_mult=sell_mult,
        amount_mult=decision_amount_mult,  # 👈 必须传我们计算出的 final_risk_mult
        today_open=today_open,             
        current_atr=latest_atr,
        holding_shares=portfolio_manager.actual_shares,
        current_avg_cost=current_avg_cost,
        highest_price=current_highest_price,
        base_shares=real_base_shares,      
        hvn_prices=latest_data.get('VAP_HVN', '[]'),  
        yesterday_close=yesterday_close
    )

    # ==========================================
    # 7. 极简实盘指挥舱 UI (高信噪比输出 + 战术心法)
    # ==========================================
    
    jump_pct = ((today_open - yesterday_close) / yesterday_close) * 100
    
    # 定义状态心法映射字典
    state_psychology = {
        'A': "下跌初期 | 心法：保持防御，只做极深位置的左侧摸底。",
        'B': "深渊极寒 | 心法：无视恐慌，启动最大火力重拳收集带血筹码！",
        'C': "疯牛主升浪 | 心法：无限死拿吃满趋势，绝对不猜顶，仅靠条件单回落止盈。",
        'D': "弱势震荡 | 心法：高频反T，开盘微弱反弹即可套现，水下再接回。",
        'E': "阴跌绞肉 | 心法：深买快卖，泥鳅战法，赚取微小波段现金流。"
    }

    print("\n" + "="*65)
    print(f" [ {args.ticker} 极简实盘中枢 | V8.6 ]  当前日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    # 1. 账户底座
    sellable_shares_today = max(0, real_actual_shares - real_base_shares)
    print("[账户底座]")
    print(f" ├─ 可用现金: ¥{real_cash:,.2f}")
    print(f" └─ 持仓结构: 总计 {real_actual_shares:,} 股 (死锁 {real_base_shares:,} | 机动 {sellable_shares_today:,})")

    # 2. 态势感知与心法
    print("\n[态势感知]")
    hmm_state_str = today_data.get('HMM_State', '未知') if pd.notna(today_data.get('HMM_State')) else '未知'
    print(f" ├─ HMM 底层判定: {hmm_state_str}")
    print(f" ├─ 乖离率(Bias) : {today_data['Bias']:.4f}")
    print(f" ├─ 波动率(ATR)  : {latest_atr:.4f}")
    
    psycho_text = state_psychology.get(final_state, "未知状态")
    print(f" └─ 最终裁决状态 : 【 {final_state} 状态 】 => {psycho_text}")
    
    # 跳空强切提示
    if abs(jump_pct) > 1.0: 
        print(f" ⚠️ 预警: 监测到开盘跳空 {jump_pct:.2f}%，引擎已进行状态劫持干预。")

    # 3. 核心作战指令 (直接对接券商填单)
    print(f"\n[作战指令] 基准开盘价: ¥{today_open:.4f}")
    
    # -- 买单输出 --
    if decision['buy_order']:
        buy_order = decision['buy_order']
        buy_diff_pct = (buy_order['price'] - today_open) / today_open * 100
        print(f" 📥 挂单买入: ¥{buy_order['price']:.4f} | 数量: {buy_order['quantity']:,} 股 (相对开盘 {buy_diff_pct:.2f}%)")
    else:
        print(f" 📥 挂单买入: 暂无符合条件的深吸点位")

   # -- 卖单输出 --
    if final_state == 'C':
        # 🌟 修复：C 状态享有绝对特权！不管引擎有没有立即生成卖单，必须强制输出条件单指南！
        atr_mult = config['tactics'].get('trailing_stop_atr_mult', 0.8)
        drop_amount = atr_mult * latest_atr
        actual_highest = max(current_highest_price, today_open)
        trailing_stop_price = actual_highest - drop_amount
        drop_pct = (drop_amount / actual_highest * 100) if actual_highest > 0 else 0
        
        print(" 📤 追踪止盈 (条件单配置):")
        print(f"    ├─  监控高点: ¥{actual_highest:.4f}")
        print(f"    ├─  触发底线: ¥{trailing_stop_price:.4f}")
        print(f"    ├─  卖出数量: 全平可用机动仓 ({sellable_shares_today:,} 股)")
        print(f"    └─  券商填单: 【回落卖出】 -> 设定高点 ¥{actual_highest:.4f} -> 回落幅度 {drop_pct:.2f}%")
        
    elif decision['sell_order']:
        # 其他状态：普通限价卖单
        sell_order = decision['sell_order']
        actual_sell = min(sell_order['quantity'], sellable_shares_today)
        
        if actual_sell > 0:
            sell_diff_pct = (sell_order['price'] - today_open) / today_open * 100
            if sell_order['price'] <= today_open:
                print(f" 📤 挂单卖出: ¥{today_open:.4f} (反T逻辑：开盘现价直接砸) | 数量: {int(actual_sell):,} 股")
            else:
                print(f" 📤 挂单卖出: ¥{sell_order['price']:.4f} | 数量: {int(actual_sell):,} 股 (相对开盘 +{sell_diff_pct:.2f}%)")
        else:
            print(f" 📤 挂单卖出: 无机动筹码可卖 (已触及底仓保护线)")
    else:
        # 其他状态且无卖单
        print(f" 📤 挂单卖出: 暂无符合条件的高抛点位 (死拿)")

    # 4. 风控物理墙
    print("\n[风控物理墙]")
    if time_decay_orders:
        decay_qty = sum(o['quantity'] for o in time_decay_orders)
        print(f" ⏰ 时间衰减: {decay_qty:,} 股已超期，请优先以现价卖出该部分筹码释放流动性！")
    print(f" 🚫 T+1 铁律: 今日新买入筹码绝对冻结。请确保今日卖单总股数 <= {sellable_shares_today:,} 股")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
