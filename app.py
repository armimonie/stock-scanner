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
    return series.rolling(window=window, min_periods=1).mean()

def safe_rolling_std(series, window):
    try:
        # min_periods=1 설정으로 데이터가 부족해도 NaN 대신 계산 시도
        return series.rolling(window=window, min_periods=1).std()
    except Exception:
        return pd.Series(np.nan, index=series.index)

def calculate_indicators(df):
    
    df_copy = df.copy()

    # V6.0: 오류 시 None 반환
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
        
        # MFI (Money Flow Index, 14일)
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

        # ----------------- V6.2: 이격도 추가 -----------------
        # J. 이격도 (20일 이동평균선 대비 종가의 비율)
        if 'MA20' in df_copy.columns:
            # 이격도 = (현재 종가 / MA20) * 100
            df_copy['Disparity'] = (df_copy['Close'] / df_copy['MA20']) * 100
        
    except Exception as e:
        # 지표 계산 중 오류 발생 시 None 반환
        return None 

    return df_copy

# -------------------------------------------------------------
# 🌟 analyze_stock 함수 (V6.2 로직 적용 - 다이버전스, 이격도 추가) 🌟
# -------------------------------------------------------------
def analyze_stock(ticker, selected_strategies):
    # 데이터 가져오기 (최근 1년 데이터)
    try:
        # Ticker 객체를 사용하여 오류를 줄여보는 시도
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="1y") 
    except Exception:
        return []

    if df.empty or len(df) < 2 or 'Close' not in df.columns:
        return []

    # 지표 계산 시도
    df_analyzed = calculate_indicators(df)
    
    # 지표 계산 실패 시(None 반환) 원본 데이터프레임을 사용 (최소한의 분석 시도)
    if df_analyzed is None:
        st.warning(f"🚨 {ticker} 지표 계산 중 오류가 발생했습니다. 일부 전략은 건너뜁니다.")
        df = df.copy() # 원본 df를 사용
    else:
        df = df_analyzed
    
    # 데이터프레임이 최소 6일 이상이어야 분석 가능 (다이버전스를 위해)
    if len(df) < 6:
        return []

    # 최신 데이터 기준
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    
    matched_reasons = []

    # ================= V6.1 기반 안정화된 타점 전략 로직 =================
    
    # 전략 A: 강력 수급 폭발 (거래량 1.5배)
    if "A. 강력 수급 폭발 (거래량 1.5배)" in selected_strategies and 'VolMA20' in df.columns:
        if not pd.isna(today.get('Volume', np.nan)) and not pd.isna(today.get('VolMA20', np.nan)) and today['Volume'] > (today['VolMA20'] * 1.5) and today['Close'] > today['Open']:
            pct_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
            matched_reasons.append({"strategy": "A. 강력 수급 폭발", "reason": f"🔥 거래량이 평소 1.5배 이상 터지며 {pct_change:.2f}% 급등했습니다. (강한 매수 유입)"})

    # 전략 B: 단기/중기 이동평균선 골든크로스 (MA20 > MA60)
    if "B. 단기/중기 이동평균선 골든크로스 (MA20 > MA60)" in selected_strategies and 'MA60' in df.columns:
        t20, t60 = today.get('MA20', np.nan), today.get('MA60', np.nan)
        y20, y60 = yesterday.get('MA20', np.nan), yesterday.get('MA60', np.nan)
        if not pd.isna(t20) and not pd.isna(t60) and not pd.isna(y20) and not pd.isna(y60) and \
           t20 > t60 and y20 <= y60:
            matched_reasons.append({"strategy": "B. 이동평균선 골든크로스", "reason": "🚀 20일선이 60일선을 상향 돌파하는 **단기/중기 추세 전환 신호** 발생."})

    # 전략 C: RSI 과매도 반등 (30 이하)
    if "C. RSI 과매도 반등 (30 이하)" in selected_strategies and 'RSI' in df.columns:
        trsi, yrsi = today.get('RSI', np.nan), yesterday.get('RSI', np.nan)
        if not pd.isna(trsi) and not pd.isna(yrsi) and \
           yrsi <= 30 and trsi > yrsi and today['Close'] > today['Open']:
            matched_reasons.append({"strategy": "C. RSI 과매도 반등", "reason": f"📈 RSI({trsi:.1f})가 30 이하 과매도 구간에서 벗어나며 **단기 강력 반등 시그널** 포착."})

    # 전략 D: MACD 시그널선 상향 돌파
    if "D. MACD 시그널선 상향 돌파" in selected_strategies and 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        tmacd = today.get('MACD', np.nan)
        tsig = today.get('MACD_Signal', np.nan)
        ymacd = yesterday.get('MACD', np.nan)
        ysig = yesterday.get('MACD_Signal', np.nan)
        
        if not pd.isna(tmacd) and not pd.isna(tsig) and not pd.isna(ymacd) and not pd.isna(ysig) and \
           tmacd > tsig and ymacd <= ysig:
            matched_reasons.append({"strategy": "D. MACD 골든크로스", "reason": "🌟 MACD선이 시그널선을 상향 돌파하며 **강력한 모멘텀 상승 신호** 발생."})

    # 전략 E: MFI 과매도 반등 (20 이하)
    if "E. MFI 과매도 반등 (20 이하)" in selected_strategies and 'MFI' in df.columns:
        tmfi, ymfi = today.get('MFI', np.nan), yesterday.get('MFI', np.nan)
        if not pd.isna(tmfi) and not pd.isna(ymfi) and \
           ymfi <= 20 and tmfi > ymfi and today['Close'] > today['Open']:
            matched_reasons.append({"strategy": "E. MFI 과매도 반등", "reason": f"💰 MFI({tmfi:.1f})가 20 이하에서 벗어나며 **단기 자금 유입 반등 시그널** 포착."})

    # 전략 F: 볼린저밴드 상단 돌파
    if "F. 볼린저밴드 상단 돌파" in selected_strategies and 'BB_Upper' in df.columns:
        tclose = today.get('Close', np.nan)
        tbbup = today.get('BB_Upper', np.nan)
        if not pd.isna(tbbup) and not pd.isna(tclose) and tclose > tbbup:
            matched_reasons.append({"strategy": "F. 볼린저밴드 상단 돌파", "reason": "⚡ 볼린저밴드 상단을 돌파하며 **강한 추세 확장 및 변동성 확대 신호** 발생."})

    # 전략 G: 장대양봉 및 짧은 꼬리 (차트 패턴 간접 반영)
    if "G. 장대양봉 및 짧은 꼬리" in selected_strategies:
        try:
            candle_range = today['High'] - today['Low']
            body_range = abs(today['Close'] - today['Open'])
            
            # 몸통 비율 70% 이상, 3% 이상 상승
            if candle_range > 0 and (body_range / candle_range) >= 0.7 and (today['Close'] / yesterday['Close'] - 1) > 0.03:
                matched_reasons.append({"strategy": "G. 장대양봉 및 짧은 꼬리", "reason": "🕯️ 몸통 비율이 70% 이상인 **3% 이상 급등 양봉 포착** (매수세 우위 확인)."})
        except Exception:
             # 데이터 불안정으로 계산 실패 시 무시
             pass
            
    # ================= V6.2: 다이버전스 및 이격도 전략 로직 추가 =================
    
    # 다이버전스 분석을 위한 최근 5일 데이터 준비 (n=5)
    n = 5
    recent_df = df.iloc[-n-1:-1].copy() # 어제까지의 n일 데이터 (총 n개)
    
    # V6.2: 다이버전스 전제 조건: 주가는 n일 동안 저점을 갱신했는가?
    # 종가 기준 저점 비교가 가장 보편적입니다.
    price_low_new = today.get('Close', np.nan) 
    price_low_old = recent_df['Close'].min()
    
    # 주가 하락 (새 저점 < 이전 n일간 저점)이 전제되어야 상승 다이버전스 검색 가능
    is_price_diverging = not pd.isna(price_low_new) and not pd.isna(price_low_old) and price_low_new < price_low_old 
    
    # 전략 H: RSI 상승 다이버전스 (RSI 저점 상승)
    if "H. RSI 상승 다이버전스" in selected_strategies and is_price_diverging and 'RSI' in df.columns:
        rsi_low_new = today.get('RSI', np.nan)
        rsi_low_old = recent_df['RSI'].min() if 'RSI' in recent_df.columns else np.nan
        
        # RSI 저점 상승 (새 저점 > 이전 저점) and RSI 40 이하에서 발생 (신뢰도 높음)
        if not pd.isna(rsi_low_new) and not pd.isna(rsi_low_old) and rsi_low_new > rsi_low_old and rsi_low_new < 40: 
            matched_reasons.append({"strategy": "H. RSI 상승 다이버전스", "reason": f"⚡️ 주가 저점 하락에도 RSI({rsi_low_new:.1f})는 상승하여 **강력한 추세 반전(다이버전스)** 신호 포착."})

    # 전략 I: MACD 상승 다이버전스 (MACD 저점 상승)
    if "I. MACD 상승 다이버전스" in selected_strategies and is_price_diverging and 'MACD' in df.columns:
        macd_low_new = today.get('MACD', np.nan)
        macd_low_old = recent_df['MACD'].min() if 'MACD' in recent_df.columns else np.nan
        
        # MACD 저점 상승 (새 저점 > 이전 저점) and MACD 0선 이하에서 발생 (신뢰도 높음)
        if not pd.isna(macd_low_new) and not pd.isna(macd_low_old) and macd_low_new > macd_low_old and today.get('MACD', 1) < 0:
            matched_reasons.append({"strategy": "I. MACD 상승 다이버전스", "reason": f"✨ 주가 하락에도 MACD({macd_low_new:.2f})는 상승하여 **중기 추세 반전(다이버전스)** 신호 포착."})

    # 전략 J: MA 이격도 과매도 (20일선 대비 95% 이하)
    if "J. MA 이격도 과매도" in selected_strategies and 'Disparity' in df.columns:
        tdisparity = today.get('Disparity', np.nan)
        # 이격도가 95% 이하: 주가가 20일 이동평균선보다 5% 이상 하락
        if not pd.isna(tdisparity) and tdisparity <= 95.0:
            matched_reasons.append({"strategy": "J. MA 이격도 과매도", "reason": f"📉 이격도({tdisparity:.1f}%)가 95% 이하로 **단기 낙폭 과대** 상태입니다. 평균 회귀 기대."})
            
    return matched_reasons

# ---------------------------------------------------------
# 2. 차트 시각화 함수 (변화 없음)
# ---------------------------------------------------------
def plot_chart(ticker, df, strategy_type, analyst_rec):
    if df is None or df.empty or 'Close' not in df.columns:
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
    ax2_vol.set_ylabel('Volume', color='gray')
    ax2_vol.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper left')
    
    # 3. MACD 차트 (ax3)
    if has_macd:
        ax3.plot(df.index, df['MACD'], label='MACD Line', color='blue')
        ax3.plot(df.index, df['MACD_Signal'], label='Signal Line', color='red')
        ax3.bar(df.index, df['MACD'] - df['MACD_Signal'], label='Histogram', color='gray', alpha=0.5)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title("MACD Indicator")
        ax3.legend(loc='upper left')


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
    
    fig = plot_chart(ticker, df, "개별 조회", analyst_rec) 
    if fig:
        st.pyplot(fig)
    else:
        st.warning(f"티커 {ticker}의 차트 데이터를 불러오거나 계산하는 데 문제가 발생했습니다.")
        
    st.markdown("---")


def main():
    st.set_page_config(page_title="AI Trading Scanner V6.2", layout="wide")
    st.title("🚀 AI 심화 분석 스캐너 (V6.2 - 다이버전스/이격도 추가 버전)")
    st.markdown("---")
    
    # --- 1️⃣ 사이드바 설정 ---
    
    st.sidebar.header("1️⃣ 개별 종목 분석")
    single_ticker = st.sidebar.text_input("티커 개별 조회 (예: 005930.KS)", "AAPL")
    
    # --- 2️⃣ 타점 전략 선택 (Multiselect) ---
    st.sidebar.header("2️⃣ 타점 전략 선택 (다중 선택 가능)")
    # V6.2: 새로운 전략 3개 추가
    all_strategies = [
        "A. 강력 수급 폭발 (거래량 1.5배)", 
        "B. 단기/중기 이동평균선 골든크로스 (MA20 > MA60)", 
        "C. RSI 과매도 반등 (30 이하)", 
        "D. MACD 시그널선 상향 돌파", 
        "E. MFI 과매도 반등 (20 이하)", 
        "F. 볼린저밴드 상단 돌파", 
        "G. 장대양봉 및 짧은 꼬리", 
        "H. RSI 상승 다이버전스",         # 🌟 NEW
        "I. MACD 상승 다이버전스",       # 🌟 NEW
        "J. MA 이격도 과매도",            # 🌟 NEW
    ]
    
    # 사용자가 이전 선택을 유지하도록 default 값 제거
    selected_strategies = st.sidebar.multiselect("원하는 타점을 모두 선택하세요 (OR 조건)", all_strategies)

    # --- 3️⃣ 스캔할 종목 목록 (코스피 하위 50 + 코스닥 상위 50 유지) ---
    st.sidebar.header("3️⃣ 스캔할 종목 목록 (총 100개)")
    
    # 코스닥 상위 50개 종목 리스트 (대형주 위주)
    kosdaq_top50 = "000210.KQ, 000660.KQ, 000880.KQ, 001120.KQ, 001390.KQ, 001550.KQ, 002170.KQ, 002200.KQ, 002270.KQ, 002320.KQ, 002360.KQ, 002390.KQ, 003380.KQ, 003550.KQ, 003560.KQ, 003620.KQ, 003650.KQ, 004140.KQ, 004720.KQ, 004830.KQ, 005180.KQ, 005880.KQ, 005930.KQ, 006400.KQ, 007680.KQ, 008770.KQ, 009190.KQ, 010060.KQ, 010120.KQ, 010140.KQ, 011070.KQ, 012280.KQ, 012450.KQ, 012750.KQ, 013420.KQ, 013640.KQ, 013700.KQ, 014990.KQ, 015350.KQ, 015760.KQ, 016600.KQ, 018000.KQ, 018260.KQ, 019550.KQ, 020660.KQ, 023590.KQ, 024740.KQ, 025680.KQ, 028080.KQ, 028300.KQ"
    
    # 코스피 하위 50개 종목 리스트 (소형주 위주)
    kospi_low50 = "000100.KS, 000180.KS, 000210.KS, 000220.KS, 000230.KS, 000300.KS, 000320.KS, 000370.KS, 000480.KS, 000500.KS, 000520.KS, 000540.KS, 000650.KS, 000670.KS, 000810.KS, 000860.KS, 000880.KS, 000950.KS, 000970.KS, 001040.KS, 001060.KS, 001070.KS, 001080.KS, 001120.KS, 001140.KS, 001210.KS, 001230.KS, 001250.KS, 001270.KS, 001380.KS, 001390.KS, 001430.KS, 001520.KS, 001550.KS, 001570.KS, 001630.KS, 001740.KS, 001780.KS, 001800.KS, 001820.KS, 001940.KS, 001950.KS, 002020.KS, 002030.KS, 002070.KS, 002170.KS, 002200.KS, 002210.KS, 002240.KS, 002270.KS"

    # 두 리스트를 합쳐서 기본값 설정
    default_tickers = kospi_low50 + ", " + kosdaq_top50
    st.sidebar.markdown("현재 **코스피 소형주 50개 + 코스닥 대형주 50개 (총 100개)**가 자동 설정되었습니다. **(수정 가능)**")
    tickers_input = st.sidebar.text_area("티커 목록 (쉼표 구분)", default_tickers) 
    
    # --- 4️⃣ 텔레그램 알림 설정 (고정 및 자동 활성화 유지) ---
    st.sidebar.header("4️⃣ 텔레그램 알림 설정 (자동)")
    tg_token = "7983927652:AAH8RRQpyJaika94NVmbmowvDIu5wHgfyWo"
    tg_chat_id = "1786596437"
    enable_alert = True 
    st.sidebar.success("✅ 텔레그램 알림이 코드로 고정/활성화되었습니다.")
    st.sidebar.markdown(f"**챗 ID:** `{tg_chat_id}`")
    
    # --- 메인 화면 로직 ---
    
    if st.button("🔍 타점 전략 스캔 시작"):
        if not selected_strategies:
            st.warning("분석할 전략을 1개 이상 선택해주세요. 🧘")
            return

        st.write(f"### 🕵️ '{', '.join(selected_strategies)}' 전략으로 총 {len(tickers_input.split(','))}개 종목을 스캔합니다...")
        
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
                    data_for_plot = yf.Ticker(ticker).history(period="1y")
                    data_for_plot = calculate_indicators(data_for_plot)
                    
                    # 매칭된 전략명을 모두 합쳐서 차트 함수에 전달 (차트 지표 표시를 위해)
                    strategy_list = [match['strategy'] for match in matched_reasons]
                    strategy_names = ", ".join(strategy_list)

                    # 차트 시각화
                    fig = plot_chart(ticker, data_for_plot, strategy_names, analyst_rec)
                    if fig:
                        st.pyplot(fig)
                        
                    # 매칭된 이유 출력
                    for match in matched_reasons:
                        st.info(f"**[{match['strategy']}]** {match['reason']}")
                        
                        # 텔레그램 전송
                        if enable_alert and tg_token and tg_chat_id:
                            msg = f"[신호 포착] 🚀 종목: {ticker} | 전략: {match['strategy']} | 이유: {match['reason']}"
                            send_telegram_msg(tg_token, tg_chat_id, msg)
                    
                    if enable_alert and tg_token and tg_chat_id:
                        st.success(f"📩 {ticker} 알림 전송 완료 (자동)")
                        
            progress_bar.progress((i + 1) / len(tickers))
        
        if found_count == 0:
            st.warning("선택한 전략에 맞는 종목을 찾지 못했습니다. 😢 시장 상황을 고려하여 **전략 선택을 줄이거나** 잠시 후 다시 시도해보세요. 🧘")
        else:
            st.success(f"총 {found_count}개의 매수 타점 종목을 찾았습니다. 🎉")

if __name__ == "__main__":
    main()
