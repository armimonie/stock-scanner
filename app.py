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

    # 지표 계산 
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
    df['BB_Upper'] = df['BB_Mid'] + (df['Close'].rolling(window=20).std() * 2)
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
    
    matched_reasons = []

    # ================= 다중 전략 로직 =================
    
    if "이동평균 골든크로스" in selected_strategies:
        if today['MA20'] > today['MA60'] and yesterday['MA20'] <= yesterday['MA60']:
            matched_reasons.append({"strategy": "이동평균 골든크로스", "reason": "📈 20일선이 60일선을 돌파했습니다. (추세 상승 전환)"})

    if "RSI 눌림목 반등" in selected_strategies:
        if today['Close'] > today['MA60'] and today['RSI'] < 50 and today['Close'] > today['Open']:
            matched_reasons.append({"strategy": "RSI 눌림목 반등", "reason": f"📉 상승 추세 중 RSI({today['RSI']:.1f})가 조정받고 양봉 발생."})

    if "MACD 골든크로스" in selected_strategies:
        if today['MACD'] > today['Signal_Line'] and yesterday['MACD'] <= yesterday['Signal_Line']:
            matched_reasons.append({"strategy": "MACD 골든크로스", "reason": "📊 MACD선이 시그널선을 상향 돌파했습니다. (매수 신호)"})

    if "볼린저 밴드 하단 터치" in selected_strategies:
        if today['Low'] <= today['BB_Lower'] * 1.02 and today['Close'] > today['Open']:
            matched_reasons.append({"strategy": "볼린저 밴드 하단 터치", "reason": "🛡️ 볼린저 밴드 하단 지지 후 반등 중입니다."})

    if "일목균형표 (전환선>기준선)" in selected_strategies:
        if today['Tenkan'] > today['Kijun'] and yesterday['Tenkan'] <= yesterday['Kijun']:
            matched_reasons.append({"strategy": "일목균형표 (전환선>기준선)", "reason": "☁️ 전환선이 기준선을 뚫고 올라갔습니다. (호전 신호)"})
            
    if "RSI 40 이하 진입" in selected_strategies:
        if today['RSI'] <= 40 and today['Close'] > today['Open']:
             matched_reasons.append({"strategy": "RSI 40 이하 진입", "reason": f"🧘 RSI({today['RSI']:.1f})가 40 이하로 떨어져 과매도 영역에 진입 후 반등."})
            
    if "대량 거래량 폭발" in selected_strategies:
        if today['Volume'] > (today['VolMA20'] * 3.0) and today['Close'] > today['Open']:
            pct_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
            matched_reasons.append({"strategy": "대량 거래량 폭발", "reason": f"🔥 거래량이 평소의 3배 이상 터지며 {pct_change:.2f}% 급등했습니다. (강력한 매수세 프록시)"})

    return matched_reasons

# ---------------------------------------------------------
# 2. 차트 시각화 함수 (완벽 포함)
# ---------------------------------------------------------
def plot_chart(ticker, result_data, strategy_type):
    df = result_data['data']
    
    # 2행 1열로 차트 영역 분할
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 상단 차트 (가격)
    ax1.plot(df.index, df['Close'], label='Close Price', color='black')
    ax1.set_title(f"{ticker} Analysis Chart ({strategy_type})", fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 전략별 보조지표 그리기 (MA, BB, 일목)
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
        
    # 매수 화살표 (최신 종가 위치)
    ax1.annotate('Buy Signal', xy=(df.index[-1], df['Close'].iloc[-1]), 
                 xytext=(df.index[-1], df['Close'].iloc[-1]*1.1),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    ax1.legend()

    # 하단 차트 (보조지표/거래량)
    if "MACD" in strategy_type:
        ax2.plot(df.index, df['MACD'], label='MACD', color='red')
        ax2.plot(df.index, df['Signal_Line'], label='Signal', color='blue')
        ax2.bar(df.index, df['MACD']-df['Signal_Line'], color='gray', alpha=0.3)
        ax2.set_title("MACD Oscillator")
    elif "RSI" in strategy_type or "눌림목" in strategy_type:
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax2.axhline(30, color='red', linestyle='--')
        ax2.axhline(70, color='blue', linestyle='--')
        ax2.axhline(40, color='orange', linestyle='--') # RSI 40선 추가
        ax2.set_title("RSI Indicator")
    else:
        ax2.bar(df.index, df['Volume'], color='gray')
        ax2.axhline(df['VolMA20'].iloc[-1] * 3, color='red', linestyle='--', label='Vol Spike Line')
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
        "RSI 40 이하 진입",
        "대량 거래량 폭발",
    ]
    selected_strategies = st.sidebar.multiselect("원하는 타점을 모두 선택하세요 (OR 조건)", all_strategies, default=["RSI 40 이하 진입", "MACD 골든크로스"])

    # --- 3️⃣ 시가총액 필터 ---
    st.sidebar.header("3️⃣ 시가총액 필터")
    min_market_cap = st.sidebar.number_input("최소 시가총액 (단위: 억 달러)", min_value=0, value=100)
    
    # --- 4️⃣ 텔레그램 알림 설정 (V2.0과 동일) ---
    st.sidebar.header("4️⃣ 텔레그램 알림 설정")
    tg_token = st.sidebar.text_input("봇 토큰 (Bot Token)", type="password")
    tg_chat_id = st.sidebar.text_input("챗 ID (Chat ID)")
    enable_alert = st.sidebar.checkbox("매수 신호 발생 시 알림 받기")

    st.markdown("---")

    if st.button("🔍 타점 분석 시작"):
        if not selected_strategies:
            st.warning("분석할 전략을 1개 이상 선택해주세요.")
            return

        st.write(f"### 🕵️ '{', '.join(selected_strategies)}' 전략으로 시장을 스캔합니다...")
        
        found_count = 0
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            
            # --- 시가총액 필터링 및 정보 가져오기 ---
            ticker_obj = yf.Ticker(ticker)
            try:
                info = ticker_obj.info
                market_cap_usd = info.get('marketCap', 0) / 1_000_000_000
                analyst_rec = info.get('recommendationKey', 'N/A')
            except:
                market_cap_usd = 0
                analyst_rec = 'N/A'
            
            if market_cap_usd < min_market_cap:
                progress_bar.progress((i + 1) / len(tickers))
                continue

            # --- 다중 전략 분석 실행 ---
            matched_reasons = analyze_stock(ticker, selected_strategies)
            
            if matched_reasons:
                found_count += 1
                
                # 화면 표시
                with st.expander(f"🔥 {ticker} - 매수 신호 포착! (총 {len(matched_reasons)}개 조건 만족)", expanded=True):
                    
                    st.markdown(f"**📈 시가총액:** 약 {market_cap_usd:,.1f} 억 달러")
                    st.markdown(f"**🗣️ 애널리스트 의견:** {analyst_rec.upper()}")
                    
                    # 각 매칭된 전략별로 정보 및 차트 표시
                    for match in matched_reasons:
                        st.info(f"**[{match['strategy']}]** {match['reason']}")
                        
                        # 차트 시각화를 위해 다시 데이터프레임 가져오기 (비효율적이지만 Streamlit 환경상 최적화)
                        analysis_result = analyze_stock(ticker, [match['strategy']]) 
                        
                        if analysis_result:
                            # plot_chart 함수에 필요한 데이터프레임을 넘겨줌
                            data_for_plot = yf.download(ticker, period="1y", progress=False)
                            fig = plot_chart(ticker, {"data": data_for_plot}, match['strategy'])
                            st.pyplot(fig)
                        
                        # 텔레그램 전송
                        if enable_alert and tg_token and tg_chat_id:
                            msg = f"[신호 포착] 🚀 종목: {ticker} | 전략: {match['strategy']} | 이유: {match['reason']}"
                            send_telegram_msg(tg_token, tg_chat_id, msg)
                    
                    if enable_alert and tg_token and tg_chat_id:
                        st.success(f"📩 {ticker} 알림 전송 완료")
                        
            progress_bar.progress((i + 1) / len(tickers))
        
        if found_count == 0:
            st.warning("현재 선택한 다중 전략과 필터 조건에 맞는 종목이 없습니다. 전략을 완화하거나 종목을 추가해보세요. 🧘")
        else:
            st.success(f"총 {found_count}개의 매수 타점 종목을 찾았습니다.")

if __name__ == "__main__":
    main()
