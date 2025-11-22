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
# 1. 데이터 분석 및 다중 전략 체크 함수 (V4.4 - 안정성 극대화)
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
        std_dev = safe_rolling_std(df_copy['Close'], 20).fillna(0) # std 계산 오류 및 NaN 처리 강화
        df_copy['BB_Upper'] = df_copy['BB_Mid'] + (std_dev * 2) 
        df_copy['BB_Lower'] = df_copy['BB_Mid'] - (std_dev * 2) 
        
        # 거래량 평균
        df_copy['VolMA20'] = safe_rolling_mean(df_copy['Volume'], 20)

        # 52주 데이터
        df_copy['52Wk_High'] = df_copy['High'].rolling(window=252).max()
        df_copy['52Wk_Low'] = df_copy['Low'].rolling(window=252).min()
        
    except Exception as e:
        # 지표 계산 실패 시 빈 데이터프레임 반환 (analyze_stock에서 처리)
        st.error(f"지표 계산 중 치명적 오류 발생: {e}")
        return pd.DataFrame()

    return df_copy

def analyze_stock(ticker, selected_strategies):
    # 데이터 가져오기 (최근 1년 데이터)
    try:
        df = yf.download(ticker, period="1y", progress=False)
    except Exception:
        return []

    if df.empty or len(df) < 120:
        return []

    df = calculate_indicators(df)
    
    # 지표 계산에 실패했을 경우 (V4.4에서 추가된 안정성 체크)
    if df.empty or 'MA5' not in df.columns:
        return []

    # 최신 데이터 기준
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    # NaN 값 체크 (계산이 제대로 안된 경우)
    if pd.isna(today['MA5']) or pd.isna(yesterday['MA5']):
         return []
         
    matched_reasons = []

    # ================= V4.1 완화된 다중 전략 로직 (유지) =================
    
    # 전략 A: 강력 수급 폭발 (2.5배 거래량)
    if "A. 강력 수급 폭발 (2.5배 거래량)" in selected_strategies:
        if today['Volume'] > (today['VolMA20'] * 2.5) and today['Close'] > today['Open']:
            pct_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
            matched_reasons.append({"strategy": "A. 강력 수급 폭발", "reason": f"🔥 거래량이 평소 2.5배 이상 터지며 {pct_change:.2f}% 급등했습니다. (강력한 매수세 프록시)"})

    # 전략 B: 단기/장기 정배열 골든크로스
    if "B. 단기/장기 정배열 골든크로스" in selected_strategies:
        if today['MA5'] > today['MA60'] and today['MA5'] > today['MA120'] and \
           yesterday['MA5'] <= yesterday['MA60'] or yesterday['MA5'] <= yesterday['MA120']:
            matched_reasons.append({"strategy": "B. 다중 정배열 골든크로스", "reason": "🚀 5일선이 60일, 120일선을 동시 돌파하며 강력한 장기 추세 전환 신호 발생."})

    # 전략 C: 매집 박스권 강한 돌파
    if "C. 매집 박스권 강한 돌파" in selected_strategies:
        box_high = df['High'].iloc[-60:-1].max()
        if today['Close'] > box_high * 1.01 and today['Volume'] > (today['VolMA20'] * 1.5):
            matched_reasons.append({"strategy": "C. 매집 박스권 강한 돌파", "reason": "🎯 60일 박스권 상단을 1.5배 거래량으로 돌파하며 매집 물량 소화."})

    # 전략 D: 52주 신고가/BB 상단 돌파
    if "D. 52주 신고가/BB 상단 돌파" in selected_strategies:
        if today['Close'] > today['52Wk_High'] * 0.995: 
            matched_reasons.append({"strategy": "D. 52주 신고가 근접", "reason": "🌟 52주 신고가 근접/돌파하며 강세 추세가 이어지는 시점."})
        if today['Close'] > today['BB_Upper']:
            matched_reasons.append({"strategy": "D. 볼린저밴드 상단 돌파", "reason": "⚡ 볼린저밴드 상단을 돌파하며 추세 확장 신호 발생."})

    # 전략 E: 단기 추세 정배열 돌파
    if "E. 단기 추세 정배열 돌파" in selected_strategies:
        if today['MA5'] > today['MA20'] > today['MA60'] and (today['Close'] / today['Open'] - 1) > 0.03:
            matched_reasons.append({"strategy": "E. 단기 추세 정배열 돌파", "reason": "🚀 5-20-60일선 정배열 상태에서 기준봉이 발생하며 추가 상승 기대."})

    # 전략 F: 장대양봉 및 짧은 꼬리
    if "F. 장대양봉 및 짧은 꼬리" in selected_strategies:
        candle_range = today['High'] - today['Low']
        body_range = abs(today['Close'] - today['Open'])
        
        if candle_range > 0 and (body_range / candle_range) >= 0.7 and (today['Close'] / yesterday['Close'] - 1) > 0.03:
            matched_reasons.append({"strategy": "F. 장대양봉 및 짧은 꼬리", "reason": "🕯️ 몸통 비율이 70% 이상인 3% 이상 급등 양봉 포착."})

    # 전략 G: RSI 40 이하 반등
    if "G. RSI 40 이하 반등" in selected_strategies:
        if today['RSI'] <= 40 and today['Close'] > today['Open']:
             matched_reasons.append({"strategy": "G. RSI 40 이하 반등", "reason": f"🧘 RSI({today['RSI']:.1f})가 40 이하로 떨어져 과매도 영역 진입 후 반등."})
            
    return matched_reasons

# ---------------------------------------------------------
# 2. 차트 시각화 함수 (V4.4 - 변화 없음)
# ---------------------------------------------------------
def plot_chart(ticker, df, strategy_type, analyst_rec):
    if 'MA5' not in df.columns:
        df = calculate_indicators(df)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(df.index, df['Close'], label='Close Price', color='black')
    ax1.plot(df.index, df['MA5'], label='MA5', color='cyan', alpha=0.7)
    ax1.plot(df.index, df['MA20'], label='MA20', color='green')
    ax1.plot(df.index, df['MA60'], label='MA60', color='orange')
    ax1.plot(df.index, df['MA120'], label='MA120', color='red', alpha=0.5)

    if 'BB_Upper' in df.columns:
        ax1.plot(df.index, df['BB_Upper'], 'g--', label='BB Upper', alpha=0.5)
        ax1.plot(df.index, df['BB_Lower'], 'r--', label='BB Lower', alpha=0.5)
        ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.05)
        
    ax1.set_title(f"{ticker} 분석 차트 (의견: {analyst_rec})", fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if 'RSI' in df.columns:
        ax2.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
        ax2.axhline(40, color='orange', linestyle='--', label='RSI 40')
        ax2.axhline(30, color='red', linestyle='--', label='RSI 30')
        ax2.set_title("RSI Indicator")
    else:
        ax2.set_title("RSI Indicator (Data Error)")

    
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
    
    fig = plot_chart(ticker, df, "개별 조회", analyst_rec)
    st.pyplot(fig)
    st.markdown("---")


def main():
    st.set_page_config(page_title="AI Trading Scanner V4.4", layout="wide")
    st.title("🚀 AI 심화 분석 스캐너 (V4.4 - 최종 안정화)")
    st.markdown("---")
    
    # --- 1️⃣ 사이드바 설정 ---
    
    st.sidebar.header("1️⃣ 개별 종목 분석")
    single_ticker = st.sidebar.text_input("티커 개별 조회 (예: 005930.KS)", "AAPL")
    
    # --- 2️⃣ 다중 전략 선택 (Multiselect) ---
    st.sidebar.header("2️⃣ 타점 전략 선택 (다중 선택 가능)")
    all_strategies = [
        "A. 강력 수급 폭발 (2.5배 거래량)",
        "B. 단기/장기 정배열 골든크로스",
        "C. 매집 박스권 강한 돌파",
        "D. 52주 신고가/BB 상단 돌파",
        "E. 단기 추세 정배열 돌파",
        "F. 장대양봉 및 짧은 꼬리",
        "G. RSI 40 이하 반등",
    ]
    selected_strategies = st.sidebar.multiselect("원하는 타점을 모두 선택하세요 (OR 조건)", all_strategies, default=["B. 단기/장기 정배열 골든크로스", "C. 매집 박스권 강한 돌파"])

    # --- 3️⃣ 스캔할 종목 목록 ---
    st.sidebar.header("3️⃣ 스캔할 종목 목록")
    # 스크린샷에 보이는 티커를 기본값으로 제공
    default_tickers = "005930.KS, 000660.KS, 207940.KS, 005490.KS, 035420.KS, 086960.KQ, 072560.KQ, 137450.KQ, 078350.KQ, 053800.KQ, 067630.KQ, 083900.KQ, 078020.KQ, 065510.KQ, 060250.KQ, 084650.KQ, 071850.KQ, 084990.KQ"
    tickers_input = st.sidebar.text_area("티커 목록 (쉼표 구분)", default_tickers) 
    
    # --- 4️⃣ 텔레그램 알림 설정 ---
    st.sidebar.header("4️⃣ 텔레그램 알림 설정")
    tg_token = st.sidebar.text_input("봇 토큰 (Bot Token)", type="password")
    tg_chat_id = st.sidebar.text_input("챗 ID (Chat ID)")
    enable_alert = st.sidebar.checkbox("매수 신호 발생 시 알림 받기")
    
    # --- 메인 화면 로직 ---
    
    if st.sidebar.button("📊 개별 종목 분석"):
        try:
            data = yf.download(single_ticker, period="1y", progress=False)
            if not data.empty and len(data) >= 120:
                data = calculate_indicators(data)
                _, _, analyst_rec = get_stock_info(single_ticker)
                display_ticker_info(single_ticker, data, analyst_rec)
            else:
                st.error(f"티커 '{single_ticker}'의 데이터가 부족하거나 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"티커 '{single_ticker}' 데이터 조회 중 오류가 발생했습니다: {e}")

    st.markdown("---")

    if st.button("🔍 타점 전략 스캔 시작"):
        if not selected_strategies:
            st.warning("분석할 전략을 1개 이상 선택해주세요.")
            return

        st.write(f"### 🕵️ '{', '.join(selected_strategies)}' 전략으로 시장을 스캔합니다...")
        
        tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
        found_count = 0
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            
            # --- 정보 가져오기 ---
            info, market_cap_usd, analyst_rec = get_stock_info(ticker)
            
            # --- 다중 전략 분석 실행 ---
            matched_reasons = analyze_stock(ticker, selected_strategies)
            
            if matched_reasons:
                found_count += 1
                
                # 화면 표시
                with st.expander(f"🔥 {ticker} - 매수 신호 포착! (총 {len(matched_reasons)}개 조건 만족)", expanded=True):
                    st.markdown(f"**📈 시가총액:** 약 {market_cap_usd:,.1f} 억 달러")
                    st.markdown(f"**🗣️ 애널리스트 의견:** **{analyst_rec.upper()}**")
                    
                    # 데이터를 다시 다운로드하고 지표 계산 (차트용)
                    data_for_plot = yf.download(ticker, period="1y", progress=False)
                    data_for_plot = calculate_indicators(data_for_plot)

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
            st.warning("현재 선택한 다중 전략에 맞는 종목이 없습니다. 시장 상황을 고려하여 **전략 선택을 줄이거나** 티커 목록을 더 추가해보세요. 🧘")
        else:
            st.success(f"총 {found_count}개의 매수 타점 종목을 찾았습니다.")

if __name__ == "__main__":
    main()
