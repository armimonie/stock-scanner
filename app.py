import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests  # 텔레그램 전송용

# ---------------------------------------------------------
# 0. 텔레그램 알림 함수 (Telegram Alert)
# ---------------------------------------------------------
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
# 1. 데이터 분석 및 전략 계산 함수
# ---------------------------------------------------------
def analyze_stock(ticker, strategy_type):
    # 데이터 가져오기 (최근 1년 데이터로 넉넉하게)
    df = yf.download(ticker, period="1y", progress=False)
    if df.empty:
        return None

    # --- 기본 지표 ---
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # --- RSI (14) ---
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- MACD (12, 26, 9) ---
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # --- 볼린저 밴드 (20, 2) ---
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    # --- 일목균형표 (전환선, 기준선) ---
    high9 = df['High'].rolling(window=9).max()
    low9 = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (high9 + low9) / 2  # 전환선

    high26 = df['High'].rolling(window=26).max()
    low26 = df['Low'].rolling(window=26).min()
    df['Kijun'] = (high26 + low26) / 2  # 기준선

    # 최신 데이터 기준 비교
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    result = {"matched": False, "reason": "", "data": df}

    # ================= 전략 로직 =================
    
    # 1. 기본: 이동평균 골든크로스
    if strategy_type == "이동평균 골든크로스":
        if today['MA20'] > today['MA60'] and yesterday['MA20'] <= yesterday['MA60']:
            result['matched'] = True
            result['reason'] = "📈 20일선이 60일선을 돌파했습니다. (추세 상승 전환)"

    # 2. RSI 눌림목
    elif strategy_type == "RSI 눌림목 반등":
        if today['Close'] > today['MA60'] and today['RSI'] < 50 and today['Close'] > today['Open']:
            result['matched'] = True
            result['reason'] = f"📉 상승 추세 중 RSI({today['RSI']:.1f})가 조정받고 양봉 발생."

    # 3. MACD 시그널 돌파
    elif strategy_type == "MACD 골든크로스":
        if today['MACD'] > today['Signal_Line'] and yesterday['MACD'] <= yesterday['Signal_Line']:
            result['matched'] = True
            result['reason'] = "📊 MACD선이 시그널선을 상향 돌파했습니다. (매수 신호)"

    # 4. 볼린저 밴드 하단 반등
    elif strategy_type == "볼린저 밴드 하단 터치":
        # 종가가 하단 밴드 근처에 있고 양봉일 때
        if today['Low'] <= today['BB_Lower'] * 1.02 and today['Close'] > today['Open']:
            result['matched'] = True
            result['reason'] = "🛡️ 볼린저 밴드 하단 지지 후 반등 중입니다."

    # 5. 일목균형표 호전
    elif strategy_type == "일목균형표 (전환선>기준선)":
        if today['Tenkan'] > today['Kijun'] and yesterday['Tenkan'] <= yesterday['Kijun']:
            result['matched'] = True
            result['reason'] = "☁️ 전환선이 기준선을 뚫고 올라갔습니다. (호전 신호)"

    return result

# ---------------------------------------------------------
# 2. 차트 시각화 함수 (전략별 맞춤 차트)
# ---------------------------------------------------------
def plot_chart(ticker, result_data, strategy_type):
    df = result_data['data']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 메인 차트 (가격)
    ax1.plot(df.index, df['Close'], label='Close', color='black')
    ax1.set_title(f"{ticker} - {strategy_type}", fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 전략별 보조지표 그리기
    if "볼린저" in strategy_type:
        ax1.plot(df.index, df['BB_Upper'], 'g--', label='Upper Band', alpha=0.5)
        ax1.plot(df.index, df['BB_Lower'], 'g--', label='Lower Band', alpha=0.5)
        ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='green', alpha=0.1)
    elif "일목균형표" in strategy_type:
        ax1.plot(df.index, df['Tenkan'], label='Tenkan (Conversion)', color='red')
        ax1.plot(df.index, df['Kijun'], label='Kijun (Base)', color='blue')
    else:
        # 기본 이평선
        ax1.plot(df.index, df['MA20'], label='MA20', color='green')
        ax1.plot(df.index, df['MA60'], label='MA60', color='orange')

    # 매수 화살표
    ax1.annotate('Buy Signal', xy=(df.index[-1], df['Close'].iloc[-1]), 
                 xytext=(df.index[-1], df['Close'].iloc[-1]*1.1),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    ax1.legend()

    # 하단 차트 (보조지표)
    if "MACD" in strategy_type:
        ax2.plot(df.index, df['MACD'], label='MACD', color='red')
        ax2.plot(df.index, df['Signal_Line'], label='Signal', color='blue')
        ax2.bar(df.index, df['MACD']-df['Signal_Line'], color='gray', alpha=0.3)
        ax2.set_title("MACD Oscillator")
    elif "RSI" in strategy_type:
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax2.axhline(30, color='red', linestyle='--')
        ax2.axhline(70, color='blue', linestyle='--')
        ax2.set_title("RSI Indicator")
    else:
        ax2.bar(df.index, df['Volume'], color='gray')
        ax2.set_title("Volume")
    
    ax2.legend()
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# 3. 메인 앱 UI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Trading Scanner Pro", layout="wide")
    st.title("🚀 나만의 AI 타점 스캐너 (Pro Ver.)")
    
    # --- 사이드바 설정 ---
    st.sidebar.header("1️⃣ 관심 종목 설정")
    default_tickers = "AAPL, TSLA, NVDA, MSFT, AMD"
    tickers = [t.strip() for t in st.sidebar.text_area("티커 입력 (쉼표 구분)", default_tickers).split(',')]

    st.sidebar.header("2️⃣ 전략(보조지표) 선택")
    strategies = [
        "이동평균 골든크로스",
        "RSI 눌림목 반등",
        "MACD 골든크로스",       # 신규 추가
        "볼린저 밴드 하단 터치",  # 신규 추가
        "일목균형표 (전환선>기준선)" # 신규 추가
    ]
    selected_strategy = st.sidebar.selectbox("타점 전략", strategies)

    st.sidebar.header("3️⃣ 텔레그램 알림 설정")
    # 실제 사용 시에는 본인의 봇 토큰과 ID를 입력해야 함
    tg_token = st.sidebar.text_input("봇 토큰 (Bot Token)", type="password")
    tg_chat_id = st.sidebar.text_input("챗 ID (Chat ID)")
    enable_alert = st.sidebar.checkbox("매수 신호 발생 시 알림 받기")

    st.markdown("---")

    if st.button("🔍 타점 분석 시작"):
        st.write(f"### 🕵️ '{selected_strategy}' 전략으로 시장을 스캔합니다...")
        
        found_count = 0
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                res = analyze_stock(ticker, selected_strategy)
                if res and res['matched']:
                    found_count += 1
                    
                    # 1. 화면 표시
                    with st.expander(f"🔥 {ticker} - 매수 신호 포착!", expanded=True):
                        st.info(f"**포착 이유:** {res['reason']}")
                        fig = plot_chart(ticker, res, selected_strategy)
                        st.pyplot(fig)
                    
                    # 2. 텔레그램 전송
                    if enable_alert and tg_token and tg_chat_id:
                        msg = f"[매수 신호 포착] 🚀\n종목: {ticker}\n전략: {selected_strategy}\n이유: {res['reason']}"
                        send_telegram_msg(tg_token, tg_chat_id, msg)
                        st.success(f"📩 {ticker} 알림 전송 완료")
                        
            except Exception as e:
                pass
            
            progress_bar.progress((i + 1) / len(tickers))
        
        if found_count == 0:
            st.warning("현재 조건에 맞는 종목이 없습니다. 관망하세요! 🧘")
        else:
            st.success(f"총 {found_count}개의 매수 타점 종목을 찾았습니다.")

if __name__ == "__main__":
    main()
