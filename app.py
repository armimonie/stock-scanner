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
def analyze_stock(ticker, selected_strategies):
    # 데이터 가져오기 (최근 1년 데이터)
    df = yf.download(ticker, period="1y", progress=False)
    if df.empty:
        return []

    # 지표 계산 (이전 코드와 동일)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 볼린저 밴드
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Lower'] = df['BB_Mid'] - (df['Close'].rolling(window=20).std() * 2)

    # 일목균형표
    high9 = df['High'].rolling(window=9).max()
    low9 = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (high9 + low9) / 2
    high26 = df['High'].rolling(window=26).max()
    low26 = df['Low'].rolling(window=26).min()
    df['Kijun'] = (high26 + low26) / 2

    # 거래량 평균
    df['VolMA20'] = df['Volume'].rolling(window=20).mean()

    # 최신 데이터 기준
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    matched_reasons = [] # 다중 매칭 결과를 담을 리스트

    # ================= 다중 전략 로직 =================
    
    # 전략: 이동평균 골든크로스
    if "이동평균 골든크로스" in selected_strategies:
        if today['MA20'] > today['MA60'] and yesterday['MA20'] <= yesterday['MA60']:
            matched_reasons.append({
                "strategy": "이동평균 골든크로스",
                "reason": "📈 20일선이 60일선을 돌파했습니다. (추세 상승 전환)"
            })

    # 전략: RSI 눌림목 반등 (RSI < 50)
    if "RSI 눌림목 반등" in selected_strategies:
        if today['Close'] > today['MA60'] and today['RSI'] < 50 and today['Close'] > today['Open']:
            matched_reasons.append({
                "strategy": "RSI 눌림목 반등",
                "reason": f"📉 상승 추세 중 RSI({today['RSI']:.1f})가 조정받고 양봉 발생."
            })

    # 전략: MACD 골든크로스
    if "MACD 골든크로스" in selected_strategies:
        if today['MACD'] > today['Signal_Line'] and yesterday['MACD'] <= yesterday['Signal_Line']:
            matched_reasons.append({
                "strategy": "MACD 골든크로스",
                "reason": "📊 MACD선이 시그널선을 상향 돌파했습니다. (매수 신호)"
            })

    # 전략: 볼린저 밴드 하단 터치
    if "볼린저 밴드 하단 터치" in selected_strategies:
        if today['Low'] <= today['BB_Lower'] * 1.02 and today['Close'] > today['Open']:
            matched_reasons.append({
                "strategy": "볼린저 밴드 하단 터치",
                "reason": "🛡️ 볼린저 밴드 하단 지지 후 반등 중입니다."
            })

    # 전략: 일목균형표 호전
    if "일목균형표 (전환선>기준선)" in selected_strategies:
        if today['Tenkan'] > today['Kijun'] and yesterday['Tenkan'] <= yesterday['Kijun']:
            matched_reasons.append({
                "strategy": "일목균형표 (전환선>기준선)",
                "reason": "☁️ 전환선이 기준선을 뚫고 올라갔습니다. (호전 신호)"
            })
            
    # [신규] 전략: RSI 40 이하 과매도 영역 진입
    if "RSI 40 이하 진입" in selected_strategies:
        if today['RSI'] <= 40 and today['Close'] > today['Open']:
             matched_reasons.append({
                "strategy": "RSI 40 이하 진입",
                "reason": f"🧘 RSI({today['RSI']:.1f})가 40 이하로 떨어져 과매도 영역에 진입 후 반등."
            })
            
    # [신규/수급] 전략: 대량 거래량 폭발 (기관/외인 매수세 프록시)
    if "대량 거래량 폭발" in selected_strategies:
        # 거래량이 평소 3배 이상 터지고 양봉일 때 (강력한 수급 유입으로 간주)
        if today['Volume'] > (today['VolMA20'] * 3.0) and today['Close'] > today['Open']:
            pct_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
            matched_reasons.append({
                "strategy": "대량 거래량 폭발",
                "reason": f"🔥 거래량이 평소의 3배 이상 터지며 {pct_change:.2f}% 급등했습니다. (강력한 매수세 프록시)"
            })


    return matched_reasons

# --- 차트 시각화 함수 (V2.0과 동일) ---
# plot_chart 함수는 동일하므로 생략합니다. (필요 시 V2.0 코드를 사용)
def plot_chart(ticker, result_data, strategy_type):
    # ... (V2.0의 plot_chart 함수 내용 복사) ...
    df = result_data['data']
    
    # ---------------------------------------------------------
    # (생략: Chart Plotting Logic from V2.0)
    # ---------------------------------------------------------
    
    # [차트 시각화] 코드는 V2.0의 plot_chart 함수 내용을 그대로 사용합니다.
    # 복사해서 V3.0 코드에 추가해 주세요.
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 상단: 가격 차트 (전략에 따라 이평선, 볼린저밴드, 일목균형표 등 표시)
    ax1.plot(df.index, df['Close'], label='Close Price', color='black')
    ax1.set_title(f"{ticker} Analysis Chart ({strategy_type})", fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 전략별 보조지표 그리기 (MACD, 볼린저, 일목은 여기에 추가)
    if "볼린저" in strategy_type:
        ax1.plot(df.index, df['BB_Lower'], 'g--', label='Lower Band', alpha=0.5)
    elif "일목균형표" in strategy_type:
        ax1.plot(df.index, df['Tenkan'], label='Tenkan (Conversion)', color='red')
        ax1.plot(df.index, df['Kijun'], label='Kijun (Base)', color='blue')
    else:
        ax1.plot(df.index, df['MA20'], label='MA20', color='green')

    # 매수 화살표
    ax1.annotate('Buy Signal', xy=(df.index[-1], df['Close'].iloc[-1]), 
                 xytext=(df.index[-1], df['Close'].iloc[-1]*1.1),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    ax1.legend()

    # 하단: 보조지표 (RSI, MACD 등)
    if "MACD" in strategy_type:
        ax2.plot(df.index, df['MACD'], label='MACD', color='red')
        ax2.plot(df.index, df['Signal_Line'], label='Signal', color='blue')
        ax2.set_title("MACD Oscillator")
    elif "RSI" in strategy_type or "눌림목" in strategy_type:
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax2.axhline(40, color='red', linestyle='--')
        ax2.set_title("RSI Indicator")
    else:
        ax2.bar(df.index, df['Volume'], color='gray')
        ax2.set_title("Volume")
    
    ax2.legend()
    plt.tight_layout()
    return fig
# ---------------------------------------------------------

# ---------------------------------------------------------
# 3. 메인 앱 UI (Streamlit)
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Trading Scanner V3.0", layout="wide")
    st.title("🚀 AI 다중 필터 타점 스캐너 (V3.0)")
    st.markdown("---")

    # --- 1️⃣ 사이드바 설정 ---
    st.sidebar.header("1️⃣ 관심 종목 설정")
    default_tickers = "AAPL, TSLA, NVDA, MSFT, AMD"
    tickers_input = st.sidebar.text_area("티커 입력 (쉼표 구분)", default_tickers)
    tickers = [t.strip() for t in tickers_input.split(',')]

    # --- 2️⃣ 다중 전략 선택 (Multiselect) ---
    st.sidebar.header("2️⃣ 타점 전략 선택 (다중 선택 가능)")
    all_strategies = [
        "이동평균 골든크로스",
        "RSI 눌림목 반등",
        "MACD 골든크로스",
        "볼린저 밴드 하단 터치",
        "일목균형표 (전환선>기준선)",
        "RSI 40 이하 진입", # 신규
        "대량 거래량 폭발",  # 신규 (기관/외인 매수세 프록시)
    ]
    selected_strategies
