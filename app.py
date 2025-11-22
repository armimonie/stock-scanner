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
# 롤링 평균/표준편차 함수를 더욱 안정적으로 수정
def safe_rolling_mean(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def safe_rolling_std(series, window):
    try:
        # min_periods=1 설정으로 데이터가 부족해도 NaN 대신 계산 시도
        return series.rolling(window=window, min_periods=1).std()
    except Exception:
        # 오류 발생 시 모든 값을 NaN으로 반환하여 안정성 확보
        return pd.Series(np.nan, index=series.index)

def calculate_indicators(df):
    
    df_copy = df.copy()

    # V5.8: calculate_indicators 전체를 try-except로 감싸서 프로그램 중단 방지
    try:
        # 이평선
        df_copy['MA5'] = safe_rolling_mean(df_copy['Close'], 5)
        df_copy['MA20'] = safe_rolling_mean(df_copy['Close'], 20)
        df_copy['MA60'] = safe_rolling_mean(df_copy['Close'], 60) 
        df_copy['MA120'] = safe_rolling_mean(df_copy['Close'], 120)
        
        # RSI (14일)
        delta = df_copy['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_copy['RSI'] = 100 - (100 / (1 + rs))
        
        # MFI (Money Flow Index, 14일) - 안정화
        typical_price = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
        money_flow = typical_price * df_copy['Volume']
        
        positive_mf = money_flow.where(typical_price.diff() > 0, 0).rolling(window=14).sum()
        negative_mf = money_flow.where(typical_price.diff() < 0, 0).rolling(window=14).sum().abs()
        
        money_ratio = positive_mf / negative_mf.replace(0, np.nan) 
        df_copy['MFI'] = 100 - (100 / (1 + money_ratio))
        
        # MACD (12, 26, 9)
        exp1 = df_copy['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_copy['Close'].ewm(span=26, adjust=False).mean()
        df_copy['MACD'] = exp1 - exp2
        df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
        
        # 볼린저 밴드
        df_copy['BB_Mid'] = safe_rolling_mean(df_copy['Close'], 20)
        std_dev = safe_rolling_std(df_copy['Close'], 20).fillna(0) 
        df_copy['BB_Upper'] = df_copy['BB_Mid'] + (std_dev * 2) 
        df_copy['BB_Lower'] = df_copy['BB_Mid'] - (std_dev * 2) 
        
        # 거래량 평균
        df_copy['VolMA20'] = safe_rolling_mean(df_copy['Volume'], 20)
        
    except Exception as e:
        # 지표 계산 중 예상치 못한 오류 발생 시 빈 데이터프레임이 아닌, 
        # 기존 df에 지표 컬럼을 추가하지 않은 상태로 반환 (분석 함수에서 NaN 체크)
        st.warning(f"지표 계산 중 오류 발생: {e}") 
        return df.copy() # 원본 df를 반환하여 최소한의 분석 시도

    return df_copy

def analyze_stock(ticker, selected_strategies):
    # V5.8: 데이터 로딩 try-except 강화
    try:
        df = yf.download(ticker, period="1y", progress=False)
    except Exception as e:
        st.error(f"데이터 로딩 실패 ({ticker}): {e}")
        return []

    if df.empty or len(df) < 120 or 'Close' not in df.columns:
        return []

    df = calculate_indicators(df)
    
    # 지표 계산 실패 시 df_copy 대신 원본 df가 반환되었을 수 있음.
    # 필수 데이터가 부족하면 분석 중단
    if len(df) < 2 or 'Close' not in df.columns or 'Volume' not in df.columns:
        return []

    # 최신 데이터 기준
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    # 필수 NaN 값 체크 
    # V5.8: 이평선이나 볼륨 평균이 NaN이면 전략A, B 등이 불가능하므로 제외
    if 'MA20' not in df.columns or pd.isna(today['MA20']) or pd.isna(yesterday['MA20']) or 'VolMA20' not in df.columns or pd.isna(today['VolMA20']):
         # MA/볼륨 지표가 계산되지 않았다면 A, B 전략을 제외한 나머지 전략만 시도 가능
         pass 
         
    matched_reasons = []

    # ================= V5.8 안정화된 타점 전략 로직 =================
    
    # 전략 A: 강력 수급 폭발 (거래량 1.5배)
    if "A. 강력 수급 폭발 (거래량 1.5배)" in selected_strategies and 'VolMA20' in df.columns:
        if not pd.isna(today['Volume']) and not pd.isna(today['VolMA20']) and today['Volume'] > (today['VolMA20'] * 1.5) and today['Close'] > today['Open']:
            pct_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
            matched_reasons.append({"strategy": "A. 강력 수급 폭발", "reason": f"🔥 거래량이 평소 1.5배 이상 터지며 {pct_change:.2f}% 급등했습니다. (강한 매수 유입)"})

    # 전략 B: 단기/중기 이동평균선 골든크로스 (MA20 > MA60)
    if "B. 단기/중기 이동평균선 골든크로스 (MA20 > MA60)" in selected_strategies and 'MA60' in df.columns:
        if not pd.isna(today['MA20']) and not pd.isna(today['MA60']) and not pd.isna(yesterday['MA20']) and not pd.isna(yesterday['MA60']) and \
           today['MA20'] > today['MA60'] and yesterday['MA20'] <= yesterday['MA60']:
            matched_reasons.append({"strategy": "B. 이동평균선 골든크로스", "reason": "🚀 20일선이 60일선을 상향 돌파하는 **단기/중기 추세 전환 신호** 발생."})

    # 전략 C: RSI 과매도 반등 (30 이하)
    if "C. RSI 과매도 반등 (30 이하)" in selected_strategies and 'RSI' in df.columns:
        if not pd.isna(today['RSI']) and not pd.isna(yesterday['RSI']) and \
           yesterday['RSI'] <= 30 and today['RSI'] > yesterday['RSI'] and today['Close'] > today['Open']:
            matched_reasons.append({"strategy": "C. RSI 과매도 반등", "reason": f"📈 RSI({today['RSI']:.1f})가 30 이하 과매도 구간에서 벗어나며 **단기 강력 반등 시그널** 포착."})

    # 전략 D: MACD 시그널선 상향 돌파
    if "D. MACD 시그널선 상향 돌파" in selected_strategies and 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        if not pd.isna(today['MACD']) and not pd.isna(today['MACD_Signal']) and not pd.isna(yesterday['MACD']) and not pd.isna(yesterday['MACD_Signal']) and \
           today['MACD'] > today['MACD_Signal'] and yesterday['MACD'] <= yesterday['MACD_Signal']:
            matched_reasons.append({"strategy": "D. MACD 골든크로스", "reason": "🌟 MACD선이 시그널선을 상향 돌파하며 **강력한 모멘텀 상승 신호** 발생."})

    # 전략 E: MFI 과매도 반등 (20 이하)
    if "E. MFI 과매도 반등 (20 이하)" in selected_strategies and 'MFI' in df.columns:
        if not pd.isna(today['MFI']) and not pd.isna(yesterday['MFI']) and \
           yesterday['MFI'] <= 20 and today['MFI'] > yesterday['MFI'] and today['Close'] > today['Open']:
            matched_reasons.append({"strategy": "E. MFI 과매도 반등", "reason": f"💰 MFI({today['MFI']:.1f})가 20 이하에서 벗어나며 **단기 자금 유입 반등 시그널** 포착."})

    # 전략 F: 볼린저밴드 상단 돌파
    if "F. 볼린저밴드 상단 돌파" in selected_strategies and 'BB_Upper' in df.columns:
        if not pd.isna(today['BB_Upper']) and today['Close'] > today['BB_Upper']:
            matched_reasons.append({"strategy": "F. 볼린저밴드 상단 돌파", "reason": "⚡ 볼린저밴드 상단을 돌파하며 **강한 추세 확장 및 변동성 확대 신호** 발생."})

    # 전략 G: 장대양봉 및 짧은 꼬리 (차트 패턴 간접 반영)
    if "G. 장대양봉 및 짧은 꼬리" in selected_strategies:
        candle_range = today['High'] - today['Low']
        body_range = abs(today['Close'] - today['Open'])
        
        # 몸통 비율 70% 이상, 3% 이상 상승
        if candle_range > 0 and (body_range / candle_range) >= 0.7 and (today['Close'] / yesterday['Close'] - 1) > 0.03:
            matched_reasons.append({"strategy": "G. 장대양봉 및 짧은 꼬리", "reason": "🕯️ 몸통 비율이 70% 이상인 **3% 이상 급등 양봉 포착** (매수세 우위 확인)."})
            
    return matched_reasons

# ---------------------------------------------------------
# 2. 차트 시각화 함수 (V5.8: 지표 컬럼 존재 여부 체크 강화)
# ---------------------------------------------------------
def plot_chart(ticker, df, strategy_type, analyst_rec):
    if df.empty or 'Close' not in df.columns:
        return None
        
    has_macd = 'MACD' in df.columns and not df['MACD'].isnull().all()
    
    if has_macd:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [4, 1, 1]})
        ax1, ax2, ax3 = axes
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax1, ax2 = axes
    
    # 1. 주가 및 이평선 차트
    ax1.plot(df.index, df['Close'], label='Close Price', color='black')
    
    # 이평선 컬럼이 존재할 때만 플롯
    if 'MA5' in df.columns:
        ax1.plot(df.index, df['MA5'], label='MA5', color='cyan', alpha=0.7)
    if 'MA20' in df.columns:
        ax1.plot(df.index, df['MA20'], label='MA20', color='green')
    if 'MA60' in df.columns:
        ax1.plot(df.index, df['MA60'], label='MA60', color='orange')
    if 'MA120' in df.columns:
        ax1.plot(df.index, df['MA120'], label='MA120', color='red', alpha=0.5)

    if 'BB_Upper' in df.columns:
        ax1.plot(df.index, df['BB_Upper'], 'g--', label='BB Upper', alpha=0.5)
        ax1.plot(df.index, df['BB_Lower'], 'r--', label='BB Lower', alpha=0.5)
        ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.05)
        
    ax1.set_title(f"{ticker} 분석 차트 (의견: {analyst_rec})", fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # 2. RSI/MFI 및 거래량 차트 (ax2)
    show_mfi = 'E.' in strategy_type or ('MFI' in df.columns and ('RSI' not in df.columns or df['RSI'].isnull().all()))
    
    if show_mfi and 'MFI' in df.columns and not df['MFI'].isnull().all():
         ax2.plot(df.index, df['MFI'], label='MFI (14)', color='brown')
         ax2.axhline(80, color='red', linestyle='--', label='MFI 80 (Overbought)')
         ax2.axhline(50, color='blue', linestyle=':', label='MFI 50')
         ax2.axhline(20, color='green', linestyle='--', label='MFI 20 (Oversold)')
         ax2.set_title("MFI Indicator")
    elif 'RSI' in df.columns and not df['RSI'].isnull().all():
         ax2.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
         ax2.axhline(70, color='red', linestyle='--', label='RSI 70 (Overbought)')
         ax2.axhline(50, color='blue', linestyle=':', label='RSI 50')
         ax2.axhline(30, color='green', linestyle='--', label='RSI 30 (Oversold)')
         ax2.set_title("RSI Indicator")
    else:
        ax2.set_title("Momentum Indicator (Data Error or Not Calculated)")

    ax2_vol = ax2.twinx()
    ax2_vol.bar(df.index, df['Volume'], color='gray', alpha=0.3, label='Volume')
    ax2_vol.set_ylabel('Volume', color='
