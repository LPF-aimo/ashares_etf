# -*- coding: utf-8 -*-
"""
ETF动态网格解套系统 V8.9 集团军完全体
HMM脑核模块 - 引入 GARCH兼容、RobustScaler、推理固化与多维强排语义
"""

import pandas as pd
import numpy as np
import os
import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

def inject_hmm_states(df, train_mode=True, model_dir=DEFAULT_MODEL_DIR):
    df = df.copy()
    
    # 🌟 优化三：剥离自相关性毒药 (特征重构)
    # 使用纯净的对数收益率替代滑动平均，避免自相关性
    df['HMM_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    # 缩短ATR基准周期，增强敏锐度
    df['HMM_ATR_Ratio'] = df['ATR'] / df['ATR'].rolling(window=20).mean()  # 从60改为20
    features = ['HMM_Return', 'HMM_ATR_Ratio', 'Bias']
    
    # 🚨 V8.9 抢救滤网：清除除以0导致的无限大
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean = df[features].dropna()
    
    if len(df_clean) < 200 and train_mode: 
        print(f"⚠️ 警告: 清洗后的有效数据仅剩 {len(df_clean)} 条，不足以训练 HMM！")
        df['HMM_State'] = np.nan
        return df
    
    model_path = os.path.join(model_dir, 'hmm_model_core.pkl')
    
    # 2. 模型训练与推理固化分离
    if train_mode:
        scaler = RobustScaler() 
        X = scaler.fit_transform(df_clean)
        
        # 🌟 优化一&二：加入"状态粘性"先验与全协方差
        model = GaussianHMM(
            n_components=5, 
            covariance_type="full",  # 👈 核心修改1：改为 full，允许多维相关性
            n_iter=2000, 
            random_state=42, 
            min_covar=1e-3,
            init_params="smc" # 👈 核心修改2：告诉模型不要自动初始化转移矩阵(t)
        )
        
        # 强制注入"镇定剂"：对角线权重极高，迫使模型更倾向于保持在当前状态
        transition_prior = np.eye(5) * 100 + 1.0  
        model.transmat_ = transition_prior / transition_prior.sum(axis=1, keepdims=True)
        
        try:
            model.fit(X)
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump({'model': model, 'scaler': scaler}, model_path)
            print(f"🧠 HMM 大脑已重新训练并固化至: {model_path}")
        except Exception as e:
            print(f"HMM模型训练失败: {str(e)}")
            df['HMM_State'] = np.nan
            return df
    else:
        if not os.path.exists(model_path):
            print(f"❌ 找不到固化的 HMM 大脑，请先运行寻优进行训练！")
            df['HMM_State'] = np.nan
            return df
        core = joblib.load(model_path)
        model = core['model']
        scaler = core['scaler']
        X = scaler.transform(df_clean)
        
    # 3. 状态预测与绝对安全打标
    states = model.predict(X)
    df.loc[df_clean.index, 'HMM_Raw_State'] = states
    
    state_stats = df.groupby('HMM_Raw_State').agg({
        'Bias': 'mean', 'HMM_Return': 'mean', 'HMM_ATR_Ratio': 'mean'
    })
    
    if len(state_stats) == 5:
        state_map = {}
        b_state = state_stats['Bias'].idxmin()
        c_state = state_stats['Bias'].idxmax()
        state_map[b_state] = 'B'
        state_map[c_state] = 'C'
        
        remaining_states = [s for s in state_stats.index if s not in [b_state, c_state]]
        
        # 🌟 优化四：语义映射的"容错装甲" - 综合动能得分
        # 我们给每个中间态打分： 动能分 = 收益率均值 - (0.2 * 波动率均值)
        def score_state(s):
            ret = state_stats.loc[s, 'HMM_Return']
            vol = state_stats.loc[s, 'HMM_ATR_Ratio']
            return ret - (0.2 * vol) # 惩罚高波动
            
        remaining_sorted = sorted(remaining_states, key=score_state)
        state_map[remaining_sorted[0]] = 'D' # 得分最低：阴跌绞肉
        state_map[remaining_sorted[1]] = 'E' # 得分居中：混沌横盘
        state_map[remaining_sorted[2]] = 'A' # 得分最高：温和上涨
            
        df['HMM_State'] = df['HMM_Raw_State'].map(state_map)
    else:
        df['HMM_State'] = df['HMM_Raw_State'].apply(lambda x: chr(ord('A') + int(x)) if pd.notnull(x) else np.nan)
        
    return df

class ETFOptimizer:
    def __init__(self):
        pass
        
    def inject_hmm_states(self, df, train_mode=True):
        """
        兼容面向对象的调用方式
        """
        return inject_hmm_states(df, train_mode=train_mode)
    
    def optimize_grid_params(self, df, current_position=None):
        if 'HMM_State' not in df.columns or df['HMM_State'].isna().all():
            return self._default_grid_params()
        
        latest_state = df['HMM_State'].iloc[-1]
        
        grid_params = {
            'B': {'grid_interval': 0.03, 'position_size': 0.15, 'trigger_threshold': 0.02}, 
            'D': {'grid_interval': 0.025, 'position_size': 0.12, 'trigger_threshold': 0.015}, 
            'E': {'grid_interval': 0.02, 'position_size': 0.1, 'trigger_threshold': 0.01},   
            'A': {'grid_interval': 0.015, 'position_size': 0.08, 'trigger_threshold': 0.008}, 
            'C': {'grid_interval': 0.01, 'position_size': 0.05, 'trigger_threshold': 0.005}   
        }
        
        return grid_params.get(latest_state, self._default_grid_params())
    
    def _default_grid_params(self):
        return {'grid_interval': 0.02, 'position_size': 0.1, 'trigger_threshold': 0.01}

def simulate_etf_trading(data):
    optimizer = ETFOptimizer()
    df_with_states = optimizer.inject_hmm_states(data, train_mode=True)
    grid_params = optimizer.optimize_grid_params(df_with_states)
    
    result = {
        'data': df_with_states,
        'current_state': df_with_states['HMM_State'].iloc[-1] if 'HMM_State' in df_with_states.columns else 'N/A',
        'grid_params': grid_params,
        'state_distribution': df_with_states['HMM_State'].value_counts().to_dict() if 'HMM_State' in df_with_states.columns else {}
    }
    return result
