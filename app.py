import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np

# --- 텔레그램 알림 함수 (변화 없음) ---
def send_telegram_msg(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        params = {'chat_id': chat_id, 'text': message}
        requests.get(url, params=params)
    except Exception as e:
        st.error(f"텔레그램 전송 실패: {e}")

# ---------------------------------------------------------
# 1. 데이터 분석 및 다중 전략 체크 함수 
# ---------------------------------------------------------
def safe_rolling_mean(series, window):
    return series.rolling(window=window).mean()

def safe_rolling_std(series, window):
    # rolling 계산 중 발생하는 오류 방지
    try:
        return series.rolling(window=window).std()
    except:
        return pd.Series(np.nan, index=series.index)

def calculate_indicators(df):
    
    # 데이터프레임 복사본 생성 (원본 보호)
    df_copy = df.copy()

    try:
        # 이평선
        df_copy['MA5'] = safe_rolling_mean(df_copy['Close'], 5)
        df_copy['MA20'] = safe_rolling_mean(df_copy['Close'], 20)
        df_copy['MA60'] = safe_rolling_mean(df_copy['Close'], 60)
        df_copy['MA120'] = safe_rolling_mean(df_copy['Close'], 120)
        
        # RSI
        delta = df_copy['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_copy['RSI'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드
        df_copy['BB_Mid'] = safe_rolling_mean(df_copy['Close'], 20)
        std_dev = safe_rolling_std(df_copy['Close'], 20).fillna(0) 
        df_copy['BB_Upper'] = df_copy['BB_Mid'] + (std_dev * 2) 
        df_copy['BB_Lower'] = df_copy['BB_Mid'] - (std_dev * 2) 
        
        # 거래량 평균
        df_copy['VolMA20'] = safe_rolling_mean(df_copy['Volume'], 20)

        # 52주 데이터 (계산은 유지하되, D 전략에서만 제외)
        df_copy['52Wk_High'] = df_copy['High'].rolling(window=252).max()
        df_copy['52Wk_Low'] = df_copy['Low'].rolling(window=252).min()
        
    except Exception as e:
        # 지표 계산 실패 시 빈 데이터프레임 반환
        return pd.DataFrame()

    return df_copy

def analyze_stock(ticker, selected_strategies):
    # 데이터 가져오기 (최근 1년 데이터)
    try:
        df = yf.download(ticker, period="1y", progress=False)
    except Exception:
        return []

    # 데이터가 비어있거나, 분석에 필요한 120일치 데이터가 없거나, 데이터프레임이 엉망이면 건너뜀
    if df.empty or len(df) < 120 or 'Close' not in df.columns:
        return []

    df = calculate_indicators(df)
    
    # 지표 계산에 실패했을 경우 (빈 DataFrame 반환 시)
    if df.empty or 'MA5' not in df.columns:
        return []

    # 최신 데이터 기준
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    # NaN 값 체크 (계산이 제대로 안된 경우)
    if pd.isna(today['MA5']) or pd.isna(yesterday['MA5']):
         return []
         
    matched_reasons = []

    # ================= V5.6 다중 전략 로직 (V5.5 로직 유지) =================
    
    # 전략 A: 강력 수급 폭발 (1
