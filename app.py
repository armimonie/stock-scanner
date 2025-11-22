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
def calculate_indicators(df):
    # 이평선
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()
    
    # RSI
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['Close'].rolling(window=20).std() * 2)
    
    # 거래량 평균
    df['VolMA20'] = df['Volume'].rolling(window=20).mean()

    # 52주 데이터
    df['52Wk_High'] = df['High'].rolling(window=252).max()
    df['52Wk_Low'] = df['Low'].rolling(window=252).min()
    
    return df

def analyze_stock(ticker, selected_strategies):
    # 데이터 가져오기 (최근 1년 데이터)
    df = yf.download(ticker, period="1y", progress=False)
    if df.empty or len(df) < 120:
        return []

    df = calculate_indicators(df)

    # 최신 데이터 기준
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    matched_reasons = []

    # ================= V4.0 다중 전략 로직 =================
    
    # 전략 A: 강력 수급 폭발 (3배 거래량)
    if "A. 강력 수급 폭발 (3배 거래량)" in selected_strategies:
        if today['Volume'] > (today['VolMA20'] * 3.0) and today['Close'] > today['Open']:
            pct_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
            matched_reasons.append({"strategy": "A. 강력 수급 폭발", "reason": f"🔥 거래량이 평소 3배 이상 터지며 {pct_change:.2f}% 급등했습니다. (강력한 매수세 프록시)"})

    # 전략 B: 단기/장기 정배열 골든크로스
    if "B. 단기/장기 정배열 골든크로스" in selected_strategies:
        # 단기선(5일)이 장기선(60일, 120일) 위로 모두 올라서는 골든크로스
        if today['MA5'] > today['MA60'] and today['MA5'] > today['MA120'] and \
           yesterday['MA5'] <= yesterday['MA60'] or yesterday['MA5'] <= yesterday['MA120']:
            matched_reasons.append({"strategy": "B. 다중 정배열 골든크로스", "reason": "🚀 5일선이 60일, 120일선을 동시 돌파하며 강력한 장기 추세 전환 신호 발생."})

    # 전략 C: 매집 박스권 강한 돌파
    if "C. 매집 박스권 강한 돌파" in selected_strategies:
        # 최근 60일 박스권 상단
        box_high = df['High'].iloc[-60:-1].max()
        # 오늘 종가가 박스권을 1% 이상 돌파 + 대량 거래량(평소 2배 이상)
        if today['Close'] > box_high * 1.01 and today['Volume'] > (today['VolMA20'] * 2.0):
            matched_reasons.append({"strategy": "C. 매집 박스권 강한 돌파", "reason": "🎯 60일 박스권 상단을 대량 거래량으로 강력하게 돌파하며 매집 물량 소화."})

    # 전략 D: 52주 신고가/BB 상단 돌파
    if "D. 52주 신고가/BB 상단 돌파" in selected_strategies:
        if today['Close'] > today['52Wk_High'] * 0.995: # 52주 신고가 근접 또는 돌파
            matched_reasons.append({"strategy": "D. 52주 신고가 근접", "reason": "🌟 52주 신고가 근접/돌파하며 강세 추세가 이어지는 시점."})
        if today['Close'] > today['BB_Upper']:
            matched_reasons.append({"strategy": "D. 볼린저밴드 상단 돌파", "reason": "⚡ 볼린저밴드 상단을 돌파하며 추세 확장 신호 발생."})

    # 전략 E: 단기 추세 정배열 돌파
    if "E. 단기 추세 정배열 돌파" in selected_strategies:
        # 5, 20, 60일선 정배열 + 오늘 장대양봉(시가 대비 종가 3% 이상 상승)
        if today['MA5'] > today['MA20'] > today['MA60'] and (today['Close'] / today['Open'] - 1) > 0.03:
            matched_reasons.append({"strategy": "E. 단기 추세 정배열 돌파", "reason": "🚀 5-20-60일선 정배열 상태에서 기준봉이 발생하며 추가 상승 기대."})

    # 전략 F: 장대양봉 및 짧은 꼬리
    if "F. 장대양봉 및 짧은 꼬리" in selected_strategies:
        # 몸통 비율이 캔들 길이의 80% 이상 (꼬리가 짧음) + 오늘 종가 > 어제 종가 5% 이상 상승
        candle_range = today['High'] - today['Low']
        body_range = abs(today['Close'] - today['Open'])
        
        if candle_range > 0 and (body_range / candle_range) >= 0.8 and (today['Close'] / yesterday['Close'] - 1) > 0.05:
            matched_reasons.append({"strategy": "F. 장대양봉 및 짧은 꼬리", "reason": "🕯️ 강한 매수세로 꼬리가 짧은 장대 양봉이 발생하여 매수 의지 강력."})

    # 전략 G: RSI 40 이하 반등
    if "G. RSI 40 이하 반등" in selected_strategies:
        if today['RSI'] <= 40 and today['Close'] > today['Open']:
             matched_reasons.append({"strategy": "G. RSI 40 이하 반등", "reason": f"🧘 RSI({today['RSI']:.1f})가 40 이하로 떨어져 과매도 영역 진입 후 반등."})
            

    return matched_reasons

# ---------------------------------------------------------
# 2. 차트 시각화 함수 (V4.0 - 다중 이평선/BB/RSI 모두 표시)
# ---------------------------------------------------------
def plot_chart(ticker, df, strategy_type, analyst_rec):
    # 필요한 지표가 계산되지 않은 경우 다시 계산
    if 'MA5' not in df.columns:
        df = calculate_indicators(df)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 상단 차트 (가격 및 이평선)
    ax1.plot(df.index, df['Close'], label='Close Price', color='black')
    ax1.plot(df.index, df['MA5'], label='MA5', color='cyan', alpha=0.7)
    ax1.plot(df.index, df['MA20'], label='MA20', color='green')
    ax1.plot(df.index, df['MA60'], label='MA60', color='orange')
    ax1.plot(df.index, df['MA120'], label='MA120', color='red', alpha=0.5)

    # 볼린저 밴드 (전략 D)
    if "볼린저밴드 상단 돌파" in strategy_type:
        ax1.plot(df.index, df['BB_Upper'], 'g--', label='BB Upper', alpha=0.5)
        ax1.plot(df.index, df['BB_Lower'], 'r--', label='BB Lower', alpha=0.5)
        ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.05)
        
    ax1.set_title(f"{ticker} 분석 차트 (의견: {analyst_rec})", fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 하단 차트 (RSI 및 거래량)
    ax2.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
    ax2.axhline(40, color='orange', linestyle='--', label='RSI 40')
    ax2.axhline(30, color='red', linestyle='--', label='RSI 30')
    ax2.set_title("RSI Indicator")
    
    # 거래량은 RSI 차트 위에 겹쳐서 투명하게 표시
    ax2_vol = ax2.twinx()
    ax2_vol.bar(df.index, df['Volume'], color='gray', alpha=0.3, label='Volume')
    ax2_vol.set_ylabel('Volume', color='gray')
    ax2_vol.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    return fig
# ---------------------------------------------------------

# ---------------------------------------------------------
# 3. 메인 앱 UI (Streamlit)
# ---------------------------------------------------------
def get_stock_info(ticker):
    """티커 정보, 마켓캡, 애널리스트 의견을 가져오는 헬퍼 함수"""
    ticker_obj = yf.Ticker(ticker)
    try:
        info = ticker_obj.info
        market_cap_usd = info.get('marketCap', 0) / 1_000_000_000
        analyst_rec = info.get('recommendationKey', 'N/A')
        return info, market_cap_usd, analyst_rec
    except:
        return {}, 0, 'N/A'

def display_ticker_info(ticker, df, analyst_rec):
    st.markdown(f"### {ticker} 상세 정보")
    st.markdown(f"**🗣️ 애널리스트 의견:** **{analyst_rec.upper()}**")
    
    # 차트 표시
    fig = plot_chart(ticker, df, "개별 조회", analyst_rec)
    st.pyplot(fig)
    st.markdown("---")


def main():
    st.set_page_config(page_title="AI Trading Scanner V4.0", layout="wide")
    st.title("🚀 AI 심화 분석 스캐너 (V4.0)")
    st.markdown("---")
    
    # --- 사이드바 설정 ---
    
    st.sidebar.header("1️⃣ 개별 종목 분석")
    single_ticker = st.sidebar.text_input("티커 개별 조회 (예: 005930.KS)", "AAPL")
    
    # --- 2️⃣ 다중 전략 선택 (Multiselect) ---
    st.sidebar.header("2️⃣ 타점 전략 선택 (다중 선택 가능)")
    all_strategies = [
        "A. 강력 수급 폭발 (3배 거래량)",
        "B. 단기/장기 정배열 골든크로스",
        "C. 매집 박스권 강한 돌파",
        "D. 52주 신고가/BB 상단 돌파",
        "E. 단기 추세 정배열 돌파",
        "F. 장대양봉 및 짧은 꼬리",
        "G. RSI 40 이하 반등",
    ]
    selected_strategies = st.sidebar.multiselect("원하는 타점을 모두 선택하세요 (OR 조건)", all_strategies, default=["B. 단기/장기 정배열 골든크로스", "C. 매집 박스권 강한 돌파"])

    # --- 3️⃣ 필터 및 종목 목록 ---
    st.sidebar.header("3️⃣ 필터 및 종목 목록")
    tickers_input = st.sidebar.text_area("스캔할 티커 목록 (쉼표 구분)", "AAPL, TSLA, NVDA, 005930.KS")
    min_market_cap = st.sidebar.number_input("최소 시가총액 (단위: 억 달러)", min_value=0, value=100)
    
    # --- 4️⃣ 텔레그램 알림 설정 (V2.0과 동일) ---
    st.sidebar.header("4️⃣ 텔레그램 알림 설정")
    tg_token = st.sidebar.text_input("봇 토큰 (Bot Token)", type="password")
    tg_chat_id = st.sidebar.text_input("챗 ID (Chat ID)")
    enable_alert = st.sidebar.checkbox("매수 신호 발생 시 알림 받기")
    
    # --- 메인 화면 로직 ---
    
    if st.sidebar.button("📊 개별 종목 분석"):
        data = yf.download(single_ticker, period="1y", progress=False)
        if not data.empty:
            _, _, analyst_rec = get_stock_info(single_ticker)
            display_ticker_info(single_ticker, data, analyst_rec)
        else:
            st.error(f"티커 '{single_ticker}'의 데이터를 찾을 수 없습니다. (한국 주식은 000000.KS 또는 .KQ 확인)")

    st.markdown("---")

    if st.button("🔍 타점 전략 스캔 시작"):
        if not selected_strategies:
            st.warning("분석할 전략을 1개 이상 선택해주세요.")
            return

        st.write(f"### 🕵️ '{', '.join(selected_strategies)}' 전략으로 시장을 스캔합니다...")
        
        tickers = [t.strip() for t in tickers_input.split(',')]
        found_count = 0
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            
            # --- 시가총액 필터링 및 정보 가져오기 ---
            _, market_cap_usd, analyst_rec = get_stock_info(ticker)
            
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
                    st.markdown(f"**🗣️ 애널리스트 의견:** **{analyst_rec.upper()}**")
                    
                    data_for_plot = yf.download(ticker, period="1y", progress=False)

                    for match in matched_reasons:
                        st.info(f"**[{match['strategy']}]** {match['reason']}")
                        
                        # 차트 시각화
                        fig = plot_chart(ticker, data_for_plot, match['strategy'], analyst_rec)
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
