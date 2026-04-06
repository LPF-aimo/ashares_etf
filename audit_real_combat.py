# -*- coding: utf-8 -*-
"""
V2.1 实盘时空模拟器 (精简字符版)
修复：移除所有Emoji，并强制隔离子进程的GBK编码崩溃问题
新增：物理写入成交记录到账户文件 (cash & actual_shares)
"""
import os
import sys
import shutil
import pandas as pd
import subprocess
import time
import argparse
import re
import json

def update_simulated_account(ticker, cash_change, share_change):
    """
    物理更新模拟账户文件 (configs/ticker_account.json)
    cash_change: 现金变化量 (正数为增加，负数为减少)
    share_change: 总持仓股数变化量 (正数为增加，负数为减少)
    """
    account_path = f'configs/{ticker}_account.json'
    try:
        # 读取现有账户文件
        with open(account_path, 'r', encoding='utf-8') as f:
            acc = json.load(f)
    except FileNotFoundError:
        print(f"    [错误] 账户文件 {account_path} 不存在！无法更新。")
        return
    except json.JSONDecodeError:
        print(f"    [错误] 账户文件 {account_path} 格式错误！无法更新。")
        return

    # 🌟 关键：同步更新现金和总持仓
    old_cash = acc['cash']
    old_shares = acc['actual_shares']

    acc['cash'] += cash_change
    acc['actual_shares'] += share_change

    # 确保数值不为负
    acc['cash'] = max(0, acc['cash'])
    acc['actual_shares'] = max(0, acc['actual_shares'])

    # 写回文件
    with open(account_path, 'w', encoding='utf-8') as f:
        json.dump(acc, f, indent=4)

    print(f"    [时空同步] 现金: {old_cash:.2f} -> {acc['cash']:.2f}, 总持仓: {old_shares} -> {acc['actual_shares']}")


def main():
    parser = argparse.ArgumentParser(description='实盘时空回溯压测系统')
    parser.add_argument('--ticker', type=str, default='512760', help='ETF代码')
    parser.add_argument('--days', type=int, default=5, help='回溯模拟的天数')
    args = parser.parse_args()

    ticker = args.ticker
    sim_days = args.days

    data_path = f'data/{ticker}_daily.csv'
    state_path = f'data/{ticker}_trade_state.json'
    account_path = f'configs/{ticker}_account.json'

    backup_data = data_path + '.backup'
    backup_state = state_path + '.backup'
    backup_account = account_path + '.backup'

    print("\n" + "="*50)
    print(f" [时空回溯引擎 V2.1] 准备对 [{ticker}] 进行 {sim_days} 天沙盘推演")
    print(" [系统锁定] 正在保护原始实盘数据...")
    
    for file_path, backup_path in [(data_path, backup_data), (state_path, backup_state), (account_path, backup_account)]:
        if os.path.exists(file_path):
            shutil.copy2(file_path, backup_path)

    try:
        full_df = pd.read_csv(backup_data)
        total_rows = len(full_df)
        
        if total_rows < sim_days + 10:
            print(f" [错误] 数据量不足，无法回溯 {sim_days} 天！")
            sys.exit(1)

        print("-" * 50)
        print(" [跃迁开始] 强行切断时间线...\n")
        
        for i in range(sim_days, 0, -1):
            target_idx = total_rows - i
            target_day_data = full_df.iloc[target_idx]
            sim_date = target_day_data['Date']
            sim_open = target_day_data['Open']  
            actual_high = target_day_data['High']  
            actual_low = target_day_data['Low']    
            
            sliced_df = full_df.iloc[:target_idx]
            sliced_df.to_csv(data_path, index=False)

            print("="*65)
            print(f" [时空坐标]: 模拟抵达 {sim_date} 早上 9:25")
            print(f" [环境注入]: 注入当日真实开盘价 ¥{sim_open:.4f}")
            print("="*65)
            
            # 🛡️ 强制子进程环境变量为 UTF-8，防止其内部输出截断
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            cmd = [sys.executable, "real_combat.py", "--ticker", ticker, "--open", str(sim_open)]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore', env=env)
            
            # 🛡️ 暴力过滤：在打印到 Windows 控制台前，抹除所有不支持的字符（包括 real_combat 里的 Emoji）
            clean_output = result.stdout.encode('gbk', 'ignore').decode('gbk')
            print(clean_output)
            
            if result.returncode != 0:
                print(f"\n [致命崩溃] {sim_date} 模拟执行失败！")
                print(result.stderr.encode('gbk', 'ignore').decode('gbk'))
                break

            # ==========================================
            # 盘后撮合雷达
            # ==========================================
            print("-" * 65)
            print(f" [盘后撮合雷达] 日终复盘 ({sim_date})")
            print(f"  |-- 实际最高价: ¥{actual_high:.4f}")
            print(f"  |-- 实际最低价: ¥{actual_low:.4f}")
            print("  " + "-"*40)

            # 🌟 修复：匹配新极简 UI 的 [挂单买入]
            buy_match = re.search(r'挂单买入:\s*[¥￥]?([0-9.]+)\s*\|\s*数量:\s*([0-9,]+)', result.stdout, re.DOTALL)
            if buy_match:
                buy_price = float(buy_match.group(1))
                buy_qty_str = buy_match.group(2)
                buy_qty_int = int(buy_qty_str.replace(',', ''))

                if buy_price >= actual_low:
                    print(f"  [+] (做多成交) 敌军进入伏击圈！成功在 ¥{buy_price:.4f} 买入 {buy_qty_str} 股！")
                    # 🌟 物理更新账户：减少现金，增加总持仓
                    buy_amount = float(buy_price) * buy_qty_int
                    update_simulated_account(ticker, -buy_amount, buy_qty_int)
                else:
                    print(f"  [-] (做多落空) 挂单价 ¥{buy_price:.4f} 未触及 (差了 ¥{actual_low - buy_price:.4f})")
            else:
                 print("  [ ] (做多静默) 今日无有效买入挂单。")

            # --- 🌟 卖出指令匹配引擎重构 (适配 V8.6 极简 UI) ---
            sell_price = None
            sell_qty_str = "0"
            sell_qty_int = 0
            order_label = ""

            # 1. 探测是否触发了 [追踪止盈] (C状态特供)
            trailing_match = re.search(r'触发底线:\s*[¥￥]?([0-9.]+)', result.stdout)
            
            # 2. 探测是否触发了 [挂单卖出] (普通限价单)
            normal_sell_match = re.search(r'挂单卖出:\s*[¥￥]?([0-9.]+)\s*\|\s*数量:\s*([0-9,]+)', result.stdout)

            if trailing_match:
                sell_price = float(trailing_match.group(1))
                # 从 UI 中提取机动仓股数
                qty_match = re.search(r'机动仓\s*\(([0-9,]+)\s*股\)', result.stdout)
                sell_qty_str = qty_match.group(1) if qty_match else "31500"
                sell_qty_int = int(sell_qty_str.replace(',', ''))
                order_label = "追踪止盈"
            elif normal_sell_match:
                sell_price = float(normal_sell_match.group(1))
                sell_qty_str = normal_sell_match.group(2)
                sell_qty_int = int(sell_qty_str.replace(',', ''))
                order_label = "限价卖出"

            # 执行撮合逻辑
            if sell_price is not None:
                if sell_price <= actual_high:
                    print(f"  [+] ({order_label}成交) 猎物已触网！成功在 ¥{sell_price:.4f} 抛出 {sell_qty_str} 股！")
                    # 🌟 物理更新账户：增加现金，减少总持仓
                    sell_amount = float(sell_price) * sell_qty_int
                    update_simulated_account(ticker, sell_amount, -sell_qty_int)
                else:
                    print(f"  [-] ({order_label}落空) 挂单价 ¥{sell_price:.4f} 未触及 (差了 ¥{sell_price - actual_high:.4f})")
            else:
                # 检查是否为死拿状态
                if '无限死拿' in result.stdout or '今日禁止止盈' in result.stdout:
                    print("  [*] (死拿锁定) 处于趋势波段中，今日放弃高抛，吃满利润！")
                else:
                    print("  [ ] (平仓静默) 今日无有效卖出信号。")

            print("-" * 65 + "\n")
            time.sleep(2)

    finally:
        print("\n" + "="*50)
        print(" [回溯结束] 正在恢复现实世界数据结构...")
        for backup_path, file_path in [(backup_data, data_path), (backup_state, state_path), (backup_account, account_path)]:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                os.remove(backup_path)
        print(" [系统解除封锁] 统帅，明早实盘环境完好无损。")

if __name__ == "__main__":
    main()