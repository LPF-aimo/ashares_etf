# -*- coding: utf-8 -*-
"""
金融终端导出文本 -> 标准 CSV 清洗器
直接读取脏文本，一键生成量化引擎可读的干净 CSV 文件
"""
import csv
import os

def clean_txt_to_csv(input_file, output_file):
    print(f"🚀 开始清洗数据文件: {input_file}")
    
    data_list = []
    
    # 1. 智能处理编码问题：金融软件导出通常是 GBK 或 UTF-8
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(input_file, 'r', encoding='gbk') as f:
            lines = f.readlines()
            
    for line in lines:
        line = line.strip()
        # 跳过空行和表头
        if not line or line.startswith('时间'):
            continue
            
        # 使用 split() 自动处理多个制表符和多余空格
        parts = line.split()
        
        # 确保列数正确 (根据同花顺的标准，通常是 11 列)
        if len(parts) >= 11:
            try:
                # 2. 拆分日期和星期 (例如: "2025-03-14,五" -> "2025-03-14")
                date_str = parts[0].split(',')[0]
                
                # 3. 数字清洗核心函数
                def clean_num(val):
                    # 处理无交易或停牌的占位符
                    if val in ['--', '-', '']:
                        return 0.0
                    # 剔除千分位逗号、加号和百分号
                    val = val.replace(',', '').replace('%', '').replace('+', '')
                    return float(val)

                # 4. 提取并映射为系统标准英文列名
                row_data = {
                    "Date": date_str,
                    "Open": clean_num(parts[1]),
                    "High": clean_num(parts[2]),
                    "Low": clean_num(parts[3]),
                    "Close": clean_num(parts[4]),
                    "ChangePct": clean_num(parts[5]),       # 涨幅 (%)
                    "AmplitudePct": clean_num(parts[6]),    # 振幅 (%)
                    "Volume": clean_num(parts[7]),          # 总手
                    "Amount": clean_num(parts[8]),          # 金额
                    "TurnoverRate": clean_num(parts[9]),    # 换手率 (%)
                    "TradeCount": clean_num(parts[10])      # 成交次数
                }
                
                data_list.append(row_data)
                
            except Exception as e:
                print(f"⚠️ 解析行出错，已跳过: {line} -> 错误: {e}")
                
    # 5. 强制按日期正序排列（极其重要：同花顺导出有时是倒序，会搞崩指标计算）
    data_list.sort(key=lambda x: x['Date'])
                
    # 6. 直接写入标准 CSV 文件
    # 确保输出的目录结构存在
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # 定义 CSV 表头顺序
    headers = ["Date", "Open", "High", "Low", "Close", "ChangePct", "AmplitudePct", "Volume", "Amount", "TurnoverRate", "TradeCount"]
    
    # 写入 CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_list)
        
    print(f"✅ 清洗完成！共成功处理 {len(data_list)} 条 K 线数据。")
    print(f"📁 已保存标准 CSV 至: {output_file}")
    
    return data_list

if __name__ == "__main__":
    # 获取当前脚本所在的目录 (data 文件夹)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 无论你在哪里运行，都会精准锁定脚本同级的 Table.txt
    INPUT_TXT = os.path.join(base_dir, "wxdh_Table.txt")
    OUTPUT_CSV = os.path.join(base_dir, "197260_train.csv")
    
    clean_txt_to_csv(INPUT_TXT, OUTPUT_CSV)