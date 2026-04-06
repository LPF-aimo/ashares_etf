# -*- coding: utf-8 -*-
"""
量化审计官 (StrategyAuditor) - V8.6 增强版
负责生成Markdown分析报告，不写入任何JSON文件
适配：支持二阶段执行结果格式
新增：HMM状态画像、参数敏感度重构、部署配置优化、表格净化、年化降本计算、HMM稳定性预警
"""
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
from tools.optimizer.engine import BacktestEngine
import json

class StrategyAuditor:
    """
    策略审计官
    生成深度分析报告，辅助人工决策
    """
    
    def __init__(self):
        pass
    
    def generate_report(self, ticker, df_results, original_df, base_config):
        """
        生成策略分析报告
        
        Args:
            ticker (str): 股票代码
            df_results (dict or pd.DataFrame): 网格搜索结果（可能是DataFrame或包含多阶段结果的字典）
            original_df (pd.DataFrame): 原始历史数据
            base_config (dict): 基础配置
            
        Returns:
            str: Markdown格式的分析报告
        """
        report_parts = []
        
        # 添加标题
        report_parts.append(f"# 📊 {ticker} 策略审计报告\n")
        report_parts.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_parts.append(f"**数据范围**: {original_df.iloc[0]['Date']} 至 {original_df.iloc[-1]['Date']}\n")
        report_parts.append("\n---\n")
        
        # 处理不同格式的结果数据
        if isinstance(df_results, dict):
            # 二阶段执行结果格式
            primary_results = df_results.get('primary_optimization', pd.DataFrame())
            sensitivity_results = df_results.get('sensitivity_analysis', pd.DataFrame())
            stress_results = df_results.get('financial_stress_test', pd.DataFrame())
            best_params = df_results.get('best_params', {})
        else:
            # 传统单阶段结果格式
            primary_results = df_results
            sensitivity_results = pd.DataFrame()
            stress_results = pd.DataFrame()
            best_params = {}
        
        # 1. 全景摘要 (Overview)
        report_parts.extend(self._generate_overview(primary_results, original_df))
        
        # 1.5. HMM状态画像 (新增)
        report_parts.extend(self._generate_hmm_profiling(original_df))
        
        # 1.6. HMM稳定性预警 (新增)
        report_parts.extend(self._generate_hmm_stability_warning(original_df))
        
        # 2. Top 10 优选矩阵
        report_parts.extend(self._generate_top_matrix(primary_results))
        
        # 3. 稳定性评分
        if not primary_results.empty:
            report_parts.extend(self._generate_stability_score(primary_results))
        
        # 4. 最差月份表现分析
        if not primary_results.empty and best_params:
            report_parts.extend(self._generate_worst_month_analysis(best_params, original_df))
        
        # 5. 滑点容忍度边界
        if not primary_results.empty and best_params:
            report_parts.extend(self._generate_slippage_tolerance(best_params, original_df))
        
        # 6. 参数敏感度分析（来自第二阶段）
        if not sensitivity_results.empty:
            report_parts.extend(self._generate_sensitivity_analysis(sensitivity_results, best_params))
        
        # 7. 财务压力测试分析（来自第二阶段）
        if not stress_results.empty:
            report_parts.extend(self._generate_financial_stress_analysis(stress_results, base_config))
        
        # 8. 最优参数深度重演 (Deep Replay)
        if not primary_results.empty and best_params:
            report_parts.extend(self._generate_deep_replay(best_params, original_df))
        
        # 9. 一键实盘部署 (Ready-to-Deploy JSON)
        if not primary_results.empty and best_params:
            report_parts.extend(self._generate_deployment_json(best_params, base_config))
        
        # 生成报告内容字符串
        report_content = "".join(report_parts)
        
        # 确保存在专门的报告存放目录
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        output_file = os.path.join(report_dir, f"audit_report_{ticker}.md")
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 审计报告已生成: {output_file}")
        
        return report_content
    
    def _generate_overview(self, df_results, original_df):
        """
        生成全景摘要
        """
        parts = ["## 📋 全景摘要\n\n"]
        
        # 处理 DataFrame 或空值情况
        if isinstance(df_results, pd.DataFrame):
            total_combinations = len(df_results)
            valid_combinations = len(df_results[df_results['total_trades'] > 0]) if not df_results.empty else 0
        else:
            total_combinations = 0
            valid_combinations = 0
        
        start_date = original_df.iloc[0]['Date']
        end_date = original_df.iloc[-1]['Date']
        
        parts.append(f"- **测试组合总数**: {total_combinations}\n")
        parts.append(f"- **有效组合数**: {valid_combinations}\n")
        parts.append(f"- **数据范围**: {start_date} 至 {end_date}\n")
        parts.append("\n---\n")
        
        return parts
    
    def _generate_hmm_profiling(self, original_df):
        """
        生成HMM状态画像 (新增)
        按HMM_State对original_df进行groupby，计算每个状态的关键指标
        """
        if 'HMM_State' not in original_df.columns:
            return ["## 🧠 HMM状态画像\n\n未找到HMM状态数据，跳过状态画像分析。\n\n---\n"]
        
        parts = ["## 🧠 HMM状态画像\n\n"]
        
        # 按HMM状态分组统计 - 修复问题：使用HMM_Return列而非错误的Close计算
        hmm_stats = original_df.groupby('HMM_State').agg({
            'Date': 'count',  # 样本天数
            'Bias': 'mean',   # 平均Bias
            'ATR': 'mean',    # 平均ATR
            'HMM_Return': 'mean'  # 使用已有的HMM_Return列，而非错误的Close计算
        }).rename(columns={'Date': '天数'})
        
        hmm_stats['天数占比'] = (hmm_stats['天数'] / len(original_df) * 100).round(2)
        hmm_stats = hmm_stats[['天数', '天数占比', 'Bias', 'ATR', 'HMM_Return']]
        hmm_stats.rename(columns={'HMM_Return': '平均日收益率'}, inplace=True)  # 修复列名
        
        # 构建表格
        parts.append("| 状态 | 天数 | 天数占比(%) | 平均Bias | 平均ATR | 平均日收益率 |\n")
        parts.append("|------|------|----------|---------|---------|------------|\n")
        
        for state in sorted(hmm_stats.index):
            stats = hmm_stats.loc[state]
            parts.append(f"| {state} | {stats['天数']} | {stats['天数占比']} | {stats['Bias']:.4f} | {stats['ATR']:.4f} | {stats['平均日收益率']:.4f} |\n")
        
        # 添加解读
        parts.append("\n**状态解读**:\n")
        for state in sorted(hmm_stats.index):
            stats = hmm_stats.loc[state]
            if stats['平均日收益率'] > 0:
                trend = "📈 上涨"
            elif stats['平均日收益率'] < 0:
                trend = "📉 下跌"
            else:
                trend = "➡️ 震荡"
            
            parts.append(f"- **{state}状态**: 平均Bias={stats['Bias']:.4f}, 平均ATR={stats['ATR']:.4f}, 趋势{trend}\n")
        
        parts.append("\n---\n")
        
        return parts
    
    def _generate_hmm_stability_warning(self, original_df):
        """
        生成HMM稳定性预警 (新增)
        计算状态跳变频率，判断模型是否过度拟合
        """
        if 'HMM_State' not in original_df.columns:
            return ["## ⚠️ HMM稳定性预警\n\n未找到HMM状态数据，跳过稳定性预警分析。\n\n---\n"]
        
        parts = ["## ⚠️ HMM稳定性预警\n\n"]
        
        # 计算状态跳变频率
        states = original_df['HMM_State'].tolist()
        transitions = 0
        total_days = len(states)
        
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                transitions += 1
        
        transition_rate = transitions / total_days if total_days > 0 else 0
        
        # 计算平均连续天数
        consecutive_counts = []
        current_count = 1
        for i in range(1, len(states)):
            if states[i] == states[i-1]:
                current_count += 1
            else:
                consecutive_counts.append(current_count)
                current_count = 1
        if current_count > 0:
            consecutive_counts.append(current_count)
        
        avg_consecutive_days = np.mean(consecutive_counts) if consecutive_counts else 0
        
        # 判断稳定性
        if transition_rate > 0.3:  # 每天超过30%概率跳变
            stability_level = "🔴 高风险"
            warning_text = "状态跳变过于频繁，模型可能在该ETF上过度拟合！"
        elif transition_rate > 0.15:  # 每天超过15%概率跳变
            stability_level = "🟡 中风险"
            warning_text = "状态跳变较为频繁，建议关注模型泛化能力。"
        else:
            stability_level = "🟢 低风险"
            warning_text = "状态转换平稳，模型稳定性良好。"
        
        parts.append(f"- **状态跳变频率**: {transition_rate:.4f} ({transition_rate*100:.2f}%)\n")
        parts.append(f"- **平均连续状态天数**: {avg_consecutive_days:.2f} 天\n")
        parts.append(f"- **状态跳变次数**: {transitions} 次\n")
        parts.append(f"- **稳定性评级**: {stability_level}\n")
        parts.append(f"- **预警**: {warning_text}\n")
        
        parts.append("\n---\n")
        
        return parts
    
    def _generate_top_matrix(self, df_results):
        """
        生成Top 10优选矩阵（动态自适应表头 + 净化逻辑）
        """
        if not isinstance(df_results, pd.DataFrame) or df_results.empty:
            return ["## 🏆 Top 10 优选矩阵\n\n暂无有效结果。\n\n---\n"]
        
        # 取前10名
        top_10 = df_results.head(10)
        
        parts = ["## 🏆 Top 10 优选矩阵\n\n"]
        
        # 定义要排除的固定列名
        exclude_columns = {
            'total_return', 'max_drawdown', 'total_trades', 'score', 'net_profit',
            'initial_cash', 'base_shares', 'fee_rate', 'min_fee'
        }
        
        # 获取所有参数列（动态扫描）
        all_columns = set(top_10.columns)
        param_columns = [col for col in all_columns if col not in exclude_columns]
        
        # 净化逻辑：移除方差为0的列
        fixed_params = {}
        variable_param_columns = []
        
        for col in param_columns:
            unique_vals = top_10[col].unique()
            if len(unique_vals) == 1:  # 值完全相同
                fixed_params[col] = unique_vals[0]
            else:
                variable_param_columns.append(col)
        
        # 按字母顺序排序参数列，使其更有序
        variable_param_columns.sort()
        
        # 显示固定参数
        if fixed_params:
            fixed_list = [f"{k}={v}" for k, v in fixed_params.items()]
            parts.append(f"**固定参数**: {', '.join(fixed_list)}\n\n")
        
        # 构建表头，增加"净利润"
        header_cols = ["排名"] + variable_param_columns + ["总收益率", "最大回撤", "交易笔数", "净利润", "Score"]
        parts.append("| " + " | ".join(header_cols) + " |\n")
        
        # 构建分隔行
        parts.append("| " + " | ".join(["---"] * len(header_cols)) + " |\n")
        
        for idx, row in top_10.iterrows():
            rank = idx + 1
            # 获取参数值
            param_values = [str(row.get(col, 'N/A')) for col in variable_param_columns]
            # 获取指标值
            total_return = f"{row['total_return']:.4f}"
            max_drawdown = f"{row['max_drawdown']:.4f}"
            total_trades = row['total_trades']
            net_profit = f"{row.get('net_profit', 0):.2f}"  # 新增提取净利润
            score = f"{row['score']:.4f}" if row['score'] != float('inf') else "∞"
            
            # 构建行数据
            row_data = [str(rank)] + param_values + [total_return, max_drawdown, str(total_trades), net_profit, score]
            parts.append("| " + " | ".join(row_data) + " |\n")
        
        parts.append("\n---\n")
        
        return parts
    
    def _generate_stability_score(self, df_results):
        """
        生成稳定性评分
        """
        parts = ["## 📈 稳定性评分\n\n"]
        
        if not isinstance(df_results, pd.DataFrame) or df_results.empty:
            parts.append("暂无数据进行稳定性分析。\n\n---\n")
            return parts
        
        # 获取Top1参数
        top1_config = df_results.iloc[0].to_dict()
        
        # 计算Top1周围的邻居表现（前10名）
        top10_configs = df_results.head(10)
        top1_score = top1_config['score']
        
        # 计算邻居平均分数和标准差
        neighbor_scores = top10_configs['score'].values
        neighbor_mean = np.mean(neighbor_scores)
        neighbor_std = np.std(neighbor_scores)
        
        # 稳定性评分计算
        stability_score = 1.0 - (neighbor_std / neighbor_mean) if neighbor_mean != 0 else 1.0
        
        # 判断是否存在过拟合风险
        if neighbor_std > top1_score * 0.1:  # 如果邻居的标准差过大
            warning_level = "🔴 高风险"
            warning_text = "⚠️ **警告**: 邻居参数表现差异巨大，可能存在过拟合风险！"
        elif neighbor_std > top1_score * 0.05:  # 中等差异
            warning_level = "🟡 中风险"
            warning_text = "⚠️ **注意**: 邻居参数表现有一定差异，需谨慎验证。"
        else:
            warning_level = "🟢 低风险"
            warning_text = "✅ **稳定**: 邻居参数表现一致，策略稳定性较好。"
        
        parts.append(f"- **Top1邻居平均分**: {neighbor_mean:.4f}\n")
        parts.append(f"- **Top1邻居标准差**: {neighbor_std:.4f}\n")
        parts.append(f"- **稳定性评分**: {stability_score:.4f}\n")
        parts.append(f"- **风险等级**: {warning_level}\n")
        parts.append(f"- **分析**: {warning_text}\n")
        
        parts.append("\n---\n")
        
        return parts
    
    def _generate_worst_month_analysis(self, top1_config, original_df):
        """
        生成最差月份表现分析
        """
        parts = ["## 📉 最差月份表现分析\n\n"]
        
        # 提取Top1配置并移除指标列
        clean_config = {}
        for key, value in top1_config.items():
            if key not in ['total_return', 'max_drawdown', 'total_trades', 'score']:
                clean_config[key] = value
        
        # 重新运行回测获取完整结果
        engine = BacktestEngine()
        full_result = engine.run(original_df, clean_config)
        
        equity_curve = full_result['equity_curve']
        trade_log = full_result['trade_log']
        dates = original_df['Date'].tolist()
        
        # 修复长度不匹配问题：确保equity_curve和dates长度一致
        # 如果equity_curve比dates长，则截取equity_curve
        if len(equity_curve) > len(dates):
            equity_curve = equity_curve[:len(dates)]
        # 如果equity_curve比dates短，则使用较短的长度
        elif len(equity_curve) < len(dates):
            dates = dates[:len(equity_curve)]
        
        # 计算30天滚动窗口收益率
        try:
            equity_series = pd.Series(equity_curve, index=dates)
            rolling_returns = []
            
            for i in range(len(equity_series) - 29):
                window_start = equity_series.iloc[i]
                window_end = equity_series.iloc[i + 29]
                period_return = (window_end - window_start) / window_start
                rolling_returns.append({
                    'start_date': equity_series.index[i],
                    'end_date': equity_series.index[i + 29],
                    'return': period_return
                })
            
            # 找到最差的30天周期
            if rolling_returns:
                worst_period = min(rolling_returns, key=lambda x: x['return'])
                
                # 提取最差期间的数据
                worst_start_idx = dates.index(worst_period['start_date'])
                worst_end_idx = dates.index(worst_period['end_date'])
                
                # 获取最差期间的权益曲线切片
                worst_start_equity_idx = worst_start_idx
                worst_end_equity_idx = worst_end_idx
                
                worst_equity_slice = equity_curve[worst_start_equity_idx:worst_end_equity_idx+1]
                
                # 计算最差期间的每日收益率
                worst_daily_returns = []
                for i in range(1, len(worst_equity_slice)):
                    if worst_equity_slice[i-1] != 0:
                        ret = (worst_equity_slice[i] - worst_equity_slice[i-1]) / worst_equity_slice[i-1]
                        worst_daily_returns.append(ret)
                    else:
                        worst_daily_returns.append(0)  # 避免除以零
                
                # 计算最差期间的胜率
                worst_win_days = sum(1 for r in worst_daily_returns if r > 0)
                worst_total_days = len(worst_daily_returns)
                worst_win_rate = worst_win_days / worst_total_days if worst_total_days > 0 else 0.0
                
                # 修复问题：使用字符串比较而非index查找 - 安全的交易次数计算
                worst_trade_count = len([t for t in trade_log if str(worst_period['start_date']) <= str(t['date']) <= str(worst_period['end_date'])])
                
                parts.append(f"- **最差30天周期**: {worst_period['start_date']} 至 {worst_period['end_date']}\n")
                parts.append(f"- **期间总收益率**: {worst_period['return']:.4f}\n")
                parts.append(f"- **期间日净值胜率**: {worst_win_rate:.4f} ({int(worst_win_rate*100)}%)\n")
                parts.append(f"- **期间交易次数**: {worst_trade_count}\n")  # 使用修复后的交易次数
                parts.append(f"- **期间最大单日跌幅**: {min(worst_daily_returns) if worst_daily_returns else 0:.4f}\n")
                parts.append(f"- **期间最大单日涨幅**: {max(worst_daily_returns) if worst_daily_returns else 0:.4f}\n")
            else:
                parts.append("数据不足，无法进行30天滚动分析。\n")
        except Exception as e:
            parts.append(f"计算最差月份分析时出错: {str(e)}\n")
        
        parts.append("\n---\n")
        
        return parts
    
    def _generate_slippage_tolerance(self, top1_config, original_df):
        """
        生成滑点容忍度边界
        """
        parts = ["## ⚖️ 滑点容忍度边界\n\n"]
        
        # 提取Top1配置并移除指标列
        clean_config = {}
        for key, value in top1_config.items():
            if key not in ['total_return', 'max_drawdown', 'total_trades', 'score']:
                clean_config[key] = value
        
        # 重新运行回测获取基础结果
        engine = BacktestEngine()
        base_result = engine.run(original_df, clean_config)
        
        # 检查回测结果格式并提取指标
        if 'metrics' in base_result:
            metrics = base_result['metrics']
            base_net_profit = metrics.get('net_profit', 0)
            base_total_trades = metrics.get('total_trades', 0)
        else:
            # 如果没有metrics键，尝试从其他地方获取
            base_net_profit = base_result.get('net_profit', 0)
            base_total_trades = base_result.get('total_trades', 0)
        
        if base_total_trades == 0:
            parts.append("该参数组无交易发生，无法计算滑点容忍度。\n\n---\n")
            return parts
        
        # 二分法寻找净利润归零点
        low_fee = 0.0
        high_fee = 1.0  # 假设最高手续费率不超过10%
        
        # 找到净利润变为负数的临界点 - 修复问题：必须有交易且净利润大于0才算容忍度及格
        while high_fee - low_fee > 0.0001:  # 精确到万分之一
            mid_fee = (low_fee + high_fee) / 2
            test_config = clean_config.copy()
            test_config['fee_rate'] = mid_fee
            
            test_result = engine.run(original_df, test_config)
            
            # 同样处理测试结果
            if 'metrics' in test_result:
                test_metrics = test_result['metrics']
                test_net_profit = test_metrics.get('net_profit', 0)
                test_total_trades = test_metrics.get('total_trades', 0)  # 新增提取交易数
            else:
                test_net_profit = test_result.get('net_profit', 0)
                test_total_trades = test_result.get('total_trades', 0)  # 新增提取交易数
            
            # 修复问题：必须有交易且净利润大于0才算容忍度及格
            if test_total_trades > 0 and test_net_profit > 0:
                low_fee = mid_fee
            else:
                high_fee = mid_fee
        
        tolerance_fee_rate = low_fee
        
        parts.append(f"- **基础净利润**: {base_net_profit:.2f}\n")
        parts.append(f"- **基础交易次数**: {base_total_trades}\n")
        parts.append(f"- **滑点容忍度边界**: 当手续费率达到 {tolerance_fee_rate:.4f} ({tolerance_fee_rate*100:.2f}%) 时，策略净利润归零\n")
        parts.append(f"- **建议安全边际**: 实际手续费率应低于 {tolerance_fee_rate*0.8:.4f} ({tolerance_fee_rate*0.8*100:.2f}%)\n")
        
        parts.append("\n---\n")
        
        return parts
    
    def _generate_sensitivity_analysis(self, sensitivity_results, best_params):
        """
        生成参数敏感度分析（重构版）
        """
        parts = ["## 🔬 参数敏感度分析\n\n"]
        
        if sensitivity_results.empty:
            parts.append("敏感度分析无有效结果。\n\n---\n")
            return parts
        
        # 获取基准分数（最佳参数的分数）
        base_score = best_params.get('score', 0) if best_params else 0
        
        if base_score == 0:
            parts.append("基准分数为0，无法计算敏感度。\n\n---\n")
            return parts
        
        # 识别哪些参数在敏感度测试中有变动
        if isinstance(sensitivity_results, pd.DataFrame) and not sensitivity_results.empty:
            # 找出在敏感度测试中发生变化的参数
            changed_params = []
            for col in sensitivity_results.columns:
                if col not in ['score', 'total_return', 'max_drawdown', 'total_trades', 'net_profit']:
                    unique_vals = sensitivity_results[col].unique()
                    if len(unique_vals) > 1:  # 该参数在敏感度测试中发生了变化
                        changed_params.append(col)
            
            if changed_params:
                parts.append("### 参数变动影响分析\n\n")
                
                for param in changed_params:
                    param_data = sensitivity_results[[param, 'score']]
                    
                    # 计算该参数对Score的影响
                    min_score = param_data['score'].min()
                    max_score = param_data['score'].max()
                    avg_score = param_data['score'].mean()
                    
                    # 计算相对于基准的变化百分比
                    min_impact_pct = ((min_score - base_score) / base_score * 100) if base_score != 0 else 0
                    max_impact_pct = ((max_score - base_score) / base_score * 100) if base_score != 0 else 0
                    avg_impact_pct = ((avg_score - base_score) / base_score * 100) if base_score != 0 else 0
                    
                    # 计算风险等级
                    abs_max_impact = max(abs(min_impact_pct), abs(max_impact_pct))
                    if abs_max_impact > 20:
                        risk_level = "🔴 高风险"
                        risk_desc = "该参数变动会导致显著的绩效波动"
                    elif abs_max_impact > 10:
                        risk_level = "🟡 中风险"
                        risk_desc = "该参数变动对绩效有一定影响"
                    else:
                        risk_level = "🟢 低风险"
                        risk_desc = "该参数变动对绩效影响较小"
                    
                    parts.append(f"**{param}**:\n")
                    parts.append(f"- 取值范围: [{param_data[param].min():.4f}, {param_data[param].max():.4f}]\n")
                    parts.append(f"- 影响范围: {min_impact_pct:+.2f}% ~ {max_impact_pct:+.2f}% (基准: {base_score:.4f})\n")
                    parts.append(f"- 平均影响: {avg_impact_pct:+.2f}%\n")
                    parts.append(f"- 风险等级: {risk_level}\n")
                    parts.append(f"- 评估: {risk_desc}\n\n")
            else:
                parts.append("敏感度测试中所有参数值保持不变。\n\n")
        else:
            parts.append("敏感度分析数据格式不正确。\n\n")
        
        parts.append("---\n")
        
        return parts
    
    def _generate_financial_stress_analysis(self, stress_results, base_config):
        """
        生成财务压力测试分析
        """
        parts = ["## 💸 财务压力测试分析\n\n"]
        
        if stress_results.empty:
            parts.append("财务压力测试无有效结果。\n\n---\n")
            return parts
        
        stress_result = stress_results.iloc[0]  # 取第一个结果
        
        # 基准现金
        base_cash = base_config['initial_cash']
        stress_cash = base_cash / 2  # 压力测试现金是基准的一半
        
        stress_score = stress_result['score']
        stress_trades = stress_result['total_trades']
        
        parts.append(f"- **基准现金**: {base_cash:.2f}\n")
        parts.append(f"- **压力测试现金**: {stress_cash:.2f}\n")
        parts.append(f"- **压力测试Score**: {stress_score:.4f}\n")
        parts.append(f"- **压力测试交易数**: {stress_trades}\n")
        
        if stress_trades == 0:
            parts.append("- **评估**: ❌ 策略在资金压力下触发逻辑死锁（无交易发生）\n")
        else:
            parts.append("- **评估**: ✅ 策略在资金压力下仍可运行\n")
        
        parts.append("\n---\n")
        
        return parts
    
    def _generate_deep_replay(self, top1_config, original_df):
        """
        生成最优参数深度重演
        """
        parts = ["## 🔍 最优参数深度重演\n\n"]
        
        # 提取Top1配置并移除指标列
        clean_config = {}
        for key, value in top1_config.items():
            if key not in ['total_return', 'max_drawdown', 'total_trades', 'score']:
                clean_config[key] = value
        
        # 重新运行回测获取完整结果
        engine = BacktestEngine()
        full_result = engine.run(original_df, clean_config)
        
        equity_curve = full_result['equity_curve']
        trade_log = full_result['trade_log']
        
        # 安全地获取指标
        if 'metrics' in full_result:
            metrics = full_result['metrics']
        else:
            # 如果没有metrics键，尝试构建基本指标
            initial_cash = clean_config.get('initial_cash', 100000)
            final_value = equity_curve[-1] if equity_curve else initial_cash
            total_return = (final_value - initial_cash) / initial_cash if initial_cash != 0 else 0
            total_trades = len([t for t in trade_log if t.get('type') in ['BUY', 'SELL']])
            
            # 计算最大回撤
            peak = 0
            max_dd = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                else:
                    dd = (peak - value) / peak
                    if dd > max_dd:
                        max_dd = dd
            max_drawdown = max_dd
            
            net_profit = final_value - initial_cash
            sharpe_ratio = 0  # 无法计算，设置为0
            
            metrics = {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'net_profit': net_profit,
                'sharpe_ratio': sharpe_ratio
            }
        
        # ---------------------------------------------------------
        # 修正：真正的每日净值胜率计算 (Daily Return Win Rate)
        # ---------------------------------------------------------
        if len(equity_curve) > 1:
            equity_series = pd.Series(equity_curve)
            daily_returns = equity_series.pct_change().dropna()
            
            # 只要总资产净值上涨，就算作"胜利日"
            win_days = (daily_returns > 0).sum()
            total_trading_days = len(daily_returns)
            win_rate = win_days / total_trading_days if total_trading_days > 0 else 0.0
            
            # 计算最大连续未创新高天数（最长解套期）
            equity_series_full = pd.Series(equity_curve)
            running_max = equity_series_full.expanding().max()
            days_since_high = 0
            max_days_since_high = 0
            for i in range(len(equity_series_full)):
                if equity_series_full.iloc[i] >= running_max.iloc[i]:
                    days_since_high = 0
                else:
                    days_since_high += 1
                    max_days_since_high = max(max_days_since_high, days_since_high)
            
            # 找到最大回撤谷底的日期
            drawdown_dates = []
            peak = equity_curve[0]
            max_drawdown_date = None
            max_drawdown_val = 0
            
            for i in range(len(equity_curve)):
                current_val = equity_curve[i]
                if current_val > peak:
                    peak = current_val
                else:
                    current_drawdown = (peak - current_val) / peak if peak != 0 else 0
                    if current_drawdown > max_drawdown_val:
                        max_drawdown_val = current_drawdown
                        max_drawdown_date = original_df.iloc[i]['Date'] if 'Date' in original_df.columns and i < len(original_df) else f"Day_{i}"
            
            # 修复夏普比率计算
            if len(daily_returns) > 1 and daily_returns.std() != 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            else:
                sharpe_ratio = 0.0
            
            # 计算年化降本幅度
            start_date = pd.to_datetime(original_df.iloc[0]['Date'])
            end_date = pd.to_datetime(original_df.iloc[-1]['Date'])
            years = (end_date - start_date).days / 365.25
            annualized_net_profit = metrics.get('net_profit', 0) / years if years > 0 else 0
            
        else:
            win_rate = 0.0
            max_days_since_high = 0
            max_drawdown_date = "N/A"
            sharpe_ratio = 0.0
            years = 0
            annualized_net_profit = 0.0
        
        parts.append(f"- **胜率**: {win_rate:.4f} ({int(win_rate*100)}%)\n")
        parts.append(f"- **最大连续未创新高天数**: {max_days_since_high} 天\n")
        parts.append(f"- **资金曲线跌入最大回撤谷底日期**: {max_drawdown_date}\n")
        parts.append(f"- **数据回测周期**: {years:.2f} 年\n")
        parts.append(f"- **详细指标**:\n")
        parts.append(f"  - 总收益率: {metrics.get('total_return', 0):.4f}\n")
        parts.append(f"  - 最大回撤: {metrics.get('max_drawdown', 0):.4f}\n")
        parts.append(f"  - 交易笔数: {metrics.get('total_trades', 0)}\n")
        parts.append(f"  - 净利润: {metrics.get('net_profit', 0):.2f}\n")
        
        # ⚠️ 修复：解套降本核算必须基于真实总持仓 (actual_shares)，而不是被冻结的底仓 (base_shares)
        # 如果获取不到 actual_shares，就退化使用 base_shares (防止除以 0)
        total_shares = clean_config.get('actual_shares', clean_config.get('base_shares', 0))
        
        if total_shares > 0:
            unit_cost_reduction = metrics.get('net_profit', 0) / total_shares
            parts.append(f"  - 预估单位降本: {unit_cost_reduction:.3f} 元/股\n")
            
            # 新增年化降本幅度
            if years > 0:
                annualized_unit_cost_reduction = unit_cost_reduction / years
                parts.append(f"  - 预估年化单位降本: {annualized_unit_cost_reduction:.3f} 元/股/年\n")
        
        parts.append(f"  - 夏普比率: {sharpe_ratio:.4f}\n")
        parts.append(f"  - 年化净利润: {annualized_net_profit:.2f} 元/年\n")
        
        parts.append("\n---\n")
        
        return parts

    def _generate_deployment_json(self, top1_config, base_config):
        """
        生成一键实盘部署的JSON配置（优化版）
        必须接收base_config参数，从merged_config中提取JSON结构
        """
        parts = ["## 🚀 一键实盘部署 (Ready-to-Deploy JSON)\n\n"]
        parts.append("以下是最优参数的实盘配置JSON，可直接用于实盘系统：\n\n")
        parts.append("    json =\n")
        
        # 合并配置：先复制基础配置，然后更新为寻优结果
        merged_config = base_config.copy()
        merged_config.update(top1_config)
        
        # 安全值处理函数
        def safe_val(v):
            return "inf" if v == float('inf') else ("-inf" if v == -float('inf') else v)
        
        # 构建实盘配置JSON，从merged_config中提取值
        deployment_config = {
            "base_shares": int(merged_config.get('base_shares', 31600) / 100) * 100,
            "base_grid_amount": safe_val(merged_config.get('base_grid', 1000.0)),
            "fee_rate": safe_val(merged_config.get('fee_rate', 0.000085)),
            "min_fee": safe_val(merged_config.get('min_fee', 1.0)),
            "adx_threshold": safe_val(merged_config.get('adx_threshold', 25)),
            "states": {
                "A": {
                    "bias_threshold": safe_val(merged_config.get('a_bias_threshold', 0.0)),
                    "buy_mult": safe_val(merged_config.get('a_buy_mult', 1.4)),
                    "sell_mult": safe_val(merged_config.get('a_sell_mult', 0.5)),
                    "amount_mult": safe_val(merged_config.get('a_amount_mult', 1.0))
                },
                "B": {
                    "bias_threshold": safe_val(merged_config.get('b_bias_threshold', -0.025)),
                    "buy_mult": safe_val(merged_config.get('b_buy_mult', 0.85)),
                    "sell_mult": safe_val(merged_config.get('b_sell_mult', float('inf'))),
                    "amount_mult": safe_val(merged_config.get('b_amount_mult', 2.5))
                },
                "C": {
                    "bias_threshold": safe_val(merged_config.get('c_bias_threshold', 0.0)),
                    "buy_mult": safe_val(merged_config.get('c_buy_mult', 1.0)),
                    "sell_mult": safe_val(merged_config.get('c_sell_mult', float('inf'))),
                    "amount_mult": safe_val(merged_config.get('c_amount_mult', 2.0))
                },
                "D": {
                    "bias_threshold": safe_val(merged_config.get('d_bias_threshold', 0.0)),
                    "buy_mult": safe_val(merged_config.get('d_buy_mult', -0.6)),
                    "sell_mult": safe_val(merged_config.get('d_sell_mult', -0.4)),
                    "amount_mult": safe_val(merged_config.get('d_amount_mult', 0.5))
                },
                "E": {
                    "bias_threshold": safe_val(merged_config.get('e_bias_threshold', 0.0)),
                    "buy_mult": safe_val(merged_config.get('e_buy_mult', 0.6)),
                    "sell_mult": safe_val(merged_config.get('e_sell_mult', 1.8)),
                    "amount_mult": safe_val(merged_config.get('e_amount_mult', 1.5))
                }
            },
            "tactics": {
                "profit_protect_pct": safe_val(merged_config.get('profit_protect_pct', 1.05)),
                "standard_sell_atr_mult": safe_val(merged_config.get('standard_sell_atr_mult', 0.8)),
                "gap_up_atr_threshold": safe_val(merged_config.get('gap_up_atr_threshold', 1.0)), # 新增这行
                "gap_down_atr_threshold": safe_val(merged_config.get('gap_down_atr_threshold', 1.0)),
                "trailing_stop_atr_mult": safe_val(merged_config.get('trailing_stop_atr_mult', 0.8)),
                "max_hold_days": int(merged_config.get('max_hold_days', 20))
            }
        }
        
        # 将配置转换为JSON字符串（格式化）
        json_str = json.dumps(deployment_config, indent=2, ensure_ascii=False)
        parts.append(json_str)
        parts.append("\n```\n\n")
        parts.append("---\n")
        return parts