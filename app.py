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
        url = f"https://api.telegram.com/bot{bot_token}/sendMessage"
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
        delta = df_copy['
