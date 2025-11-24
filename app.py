import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
import time
import locale

# 한국어 통화 형식을 사용하도록 설정 (숫자 포맷팅을 위해)
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except locale.Error:
    try:
        # 환경에 따라 다른 인코딩 사용
        locale.setlocale(locale.LC_ALL, 'Korean_Korea.949')
    except locale.Error:
        # 설정 실패 시 기본값 사용 (숫자 포맷팅을 수동으로 처리해야 할 수 있음)
        pass


# --- 텔레그램 알림 함수 (HTML 포맷 지정 및 안정화) ---
def send_telegram_msg(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        #st.warning("텔레그램 토큰 또는 Chat ID가 설정되지 않았습니다.")
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        params = {
            'chat_id': chat_id, 
            'text': message,
            'parse_mode': 'HTML'  # HTML 태그를 사용하도록 설정
        }
        # API 호출 시 지연 시간 추가 (너무 빠른 요청 방지)
        time.sleep(0.5) 
        response = requests.get(url, params=params)
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
    except requests.exceptions.HTTPError as e:
        # 텔레그램 API에서 발생하는 오류 (예: Chat ID 오류, 권한 오류 등)
        print(f"🚨 텔레그램 API HTTP 오류: {response.text}")
        print(f"토큰: {bot_token[:10]}... ID: {chat_id}")
    except Exception as e:
        print(f"🚨 텔레그램 전송 실패 (일반 오류): {e}")

# ---------------------------------------------------------
# 1. 데이터 분석 및 다중 전략 체크 함수
# ---------------------------------------------------------
def safe_rolling_mean(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def safe_rolling_std(series, window):
    try:
        return series.rolling(window=window, min_periods=1).std()
    except Exception:
        return pd.Series(np.nan, index=series.index)

def calculate_indicators(df):
    
    df_copy = df.copy()

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
        # 0으로 나누는 오류 방지를 위해 replace 사용
        rs = gain / loss.replace(0, np.nan) 
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

        # 이격도
        if 'MA20' in df_copy.columns and not df_copy['MA20'].isnull().all():
            df_copy['Disparity'] = (df_copy['Close'] / df_copy['MA20']) * 100
        else:
            df_copy['Disparity'] = np.nan
        
    except Exception as e:
        print(f"지표 계산 오류 발생: {e}")
        return None 

    return df_copy

# -------------------------------------------------------------
# 🌟 analyze_stock 함수 (analyze_stock은 이제 df_analyzed까지 반환합니다) 🌟 
# -------------------------------------------------------------
def analyze_stock(ticker, selected_strategies):
    # 데이터 가져오기 (최근 1년 데이터)
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="1y") 
    except Exception:
        return [], None

    if df.empty or len(df) < 2 or 'Close' not in df.columns:
        return [], None

    df_analyzed = calculate_indicators(df)
    
    if df_analyzed is None:
        df_analyzed = df.copy() # 원본 df를 사용 (지표 계산 실패)
    
    # 데이터프레임이 최소 6일 이상이어야 분석 가능 (다이버전스를 위해)
    if len(df_analyzed) < 6:
        return [], df_analyzed

    # 최신 데이터 기준
    today = df_analyzed.iloc[-1]
    yesterday = df_analyzed.iloc[-2]
    
    matched_reasons = []

    # ================= 타점 전략 로직 =================
    
    # 전략 A: 강력 수급 폭발 (거래량 1.5배)
    if "A. 강력 수급 폭발 (거래량 1.5배)" in selected_strategies and 'VolMA20' in df_analyzed.columns:
        if not pd.isna(today.get('Volume', np.nan)) and not pd.isna(today.get('VolMA20', np.nan)) and today['Volume'] > (today['VolMA20'] * 1.5) and today['Close'] > today['Open']:
            pct_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
            matched_reasons.append({"strategy": "A. 강력 수급 폭발", "reason": f"🔥 거래량이 평소 1.5배 이상 터지며 {pct_change:.2f}% 급등했습니다. (강한 매수 유입)"})

    # 전략 B: 단기/중기 이동평균선 골든크로스 (MA20 > MA60)
    if "B. 단기/중기 이동평균선 골든크로스 (MA20 > MA60)" in selected_strategies and 'MA60' in df_analyzed.columns:
        t20, t60 = today.get('MA20', np.nan), today.get('MA60', np.nan)
        y20, y60 = yesterday.get('MA20', np.nan), yesterday.get('MA60', np.nan)
        if not pd.isna(t20) and not pd.isna(t60) and not pd.isna(y20) and not pd.isna(y60) and \
           t20 > t60 and y20 <= y60:
            matched_reasons.append({"strategy": "B. 이동평균선 골든크로스", "reason": "🚀 20일선이 60일선을 상향 돌파하는 **단기/중기 추세 전환 신호** 발생."})

    # 전략 C: RSI 과매도 반등 (30 이하)
    if "C. RSI 과매도 반등 (30 이하)" in selected_strategies and 'RSI' in df_analyzed.columns:
        trsi, yrsi = today.get('RSI', np.nan), yesterday.get('RSI', np.nan)
        if not pd.isna(trsi) and not pd.isna(yrsi) and \
           yrsi <= 30 and trsi > yrsi and today['Close'] > today['Open']:
            matched_reasons.append({"strategy": "C. RSI 과매도 반등", "reason": f"📈 RSI({trsi:.1f})가 30 이하 과매도 구간에서 벗어나며 **단기 강력 반등 시그널** 포착."})

    # 전략 D: MACD 시그널선 상향 돌파
    if "D. MACD 시그널선 상향 돌파" in selected_strategies and 'MACD' in df_analyzed.columns and 'MACD_Signal' in df_analyzed.columns:
        tmacd = today.get('MACD', np.nan)
        tsig = today.get('MACD_Signal', np.nan)
        ymacd = yesterday.get('MACD', np.nan)
        ysig = yesterday.get('MACD_Signal', np.nan)
        
        if not pd.isna(tmacd) and not pd.isna(tsig) and not pd.isna(ymacd) and not pd.isna(ysig) and \
           tmacd > tsig and ymacd <= ysig:
            matched_reasons.append({"strategy": "D. MACD 골든크로스", "reason": "🌟 MACD선이 시그널선을 상향 돌파하며 **강력한 모멘텀 상승 신호** 발생."})

    # 전략 E: MFI 과매도 반등 (20 이하)
    if "E. MFI 과매도 반등 (20 이하)" in selected_strategies and 'MFI' in df_analyzed.columns:
        tmfi, ymfi = today.get('MFI', np.nan), yesterday.get('MFI', np.nan)
        if not pd.isna(tmfi) and not pd.isna(ymfi) and \
           ymfi <= 20 and tmfi > ymfi and today['Close'] > today['Open']:
            matched_reasons.append({"strategy": "E. MFI 과매도 반등", "reason": f"💰 MFI({tmfi:.1f})가 20 이하에서 벗어나며 **단기 자금 유입 반등 시그널** 포착."})

    # 전략 F: 볼린저밴드 상단 돌파
    if "F. 볼린저밴드 상단 돌파" in selected_strategies and 'BB_Upper' in df_analyzed.columns:
        tclose = today.get('Close', np.nan)
        tbbup = today.get('BB_Upper', np.nan)
        if not pd.isna(tbbup) and not pd.isna(tclose) and tclose > tbbup:
            matched_reasons.append({"strategy": "F. 볼린저밴드 상단 돌파", "reason": "⚡ 볼린저밴드 상단을 돌파하며 **강한 추세 확장 및 변동성 확대 신호** 발생."})

    # 전략 G: 장대양봉 및 짧은 꼬리 (차트 패턴 간접 반영)
    if "G. 장대양봉 및 짧은 꼬리" in selected_strategies:
        try:
            candle_range = today['High'] - today['Low']
            body_range = abs(today['Close'] - today['Open'])
            
            if candle_range > 0 and (body_range / candle_range) >= 0.7 and (today['Close'] / yesterday['Close'] - 1) > 0.03:
                matched_reasons.append({"strategy": "G. 장대양봉 및 짧은 꼬리", "reason": "🕯️ 몸통 비율이 70% 이상인 **3% 이상 급등 양봉 포착** (매수세 우위 확인)."})
        except Exception:
             pass
            
    # ================= 다이버전스 및 이격도 전략 로직 =================
    
    n = 5
    recent_df_full = df_analyzed.iloc[-(n+1):] 
    
    if len(recent_df_full) >= 2:
        today_data = recent_df_full.iloc[-1]
        recent_df = recent_df_full.iloc[:-1] 
    else:
        return matched_reasons, df_analyzed # 데이터 부족

    # V6.2: 다이버전스 전제 조건: 주가는 n일 동안 저점을 갱신했는가?
    price_low_new = today_data.get('Close', np.nan) 
    price_low_old = recent_df['Close'].min()
    
    is_price_diverging = not pd.isna(price_low_new) and not pd.isna(price_low_old) and price_low_new < price_low_old 
    
    # 전략 H: RSI 상승 다이버전스 (RSI 저점 상승)
    if "H. RSI 상승 다이버전스" in selected_strategies and is_price_diverging and 'RSI' in df_analyzed.columns:
        rsi_low_new = today_data.get('RSI', np.nan)
        rsi_low_old = recent_df['RSI'].min() if 'RSI' in recent_df.columns else np.nan
        
        if not pd.isna(rsi_low_new) and not pd.isna(rsi_low_old) and rsi_low_new > rsi_low_old and rsi_low_new < 40: 
            matched_reasons.append({"strategy": "H. RSI 상승 다이버전스", "reason": f"⚡️ 주가 저점 하락에도 RSI({rsi_low_new:.1f})는 상승하여 **강력한 추세 반전(다이버전스)** 신호 포착."})

    # 전략 I: MACD 상승 다이버전스 (MACD 저점 상승)
    if "I. MACD 상승 다이버전스" in selected_strategies and is_price_diverging and 'MACD' in df_analyzed.columns:
        macd_low_new = today_data.get('MACD', np.nan)
        macd_low_old = recent_df['MACD'].min() if 'MACD' in recent_df.columns else np.nan
        
        if not pd.isna(macd_low_new) and not pd.isna(macd_low_old) and macd_low_new > macd_low_old and today_data.get('MACD', 1) < 0:
            matched_reasons.append({"strategy": "I. MACD 상승 다이버전스", "reason": f"✨ 주가 하락에도 MACD({macd_low_new:.2f})는 상승하여 **중기 추세 반전(다이버전스)** 신호 포착."})

    # 전략 J: MA 이격도 과매도 (20일선 대비 95% 이하)
    if "J. MA 이격도 과매도" in selected_strategies and 'Disparity' in df_analyzed.columns:
        tdisparity = today_data.get('Disparity', np.nan)
        if not pd.isna(tdisparity) and tdisparity <= 95.0:
            matched_reasons.append({"strategy": "J. MA 이격도 과매도", "reason": f"📉 이격도({tdisparity:.1f}%)가 95% 이하로 **단기 낙폭 과대** 상태입니다. 평균 회귀 기대."})
            
    return matched_reasons, df_analyzed

# ---------------------------------------------------------
# 2. 차트 시각화 함수
# ---------------------------------------------------------
def plot_chart(ticker, df, strategy_type, analyst_rec):
    if df is None or df.empty or 'Close' not in df.columns:
        return None
        
    has_macd = 'MACD' in df.columns and not df['MACD'].isnull().all()
    show_momentum = ('RSI' in df.columns and not df['RSI'].isnull().all()) or \
                    ('MFI' in df.columns and not df['MFI'].isnull().all()) or \
                    ('Volume' in df.columns and not df['Volume'].isnull().all())
    
    num_subcharts = 1
    if show_momentum: num_subcharts += 1
    if has_macd: num_subcharts += 1
        
    # Gridspec 설정
    if num_subcharts == 3:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [4, 1, 1]})
        ax1, ax2, ax3 = axes
    elif num_subcharts == 2:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        axes = [ax1]
    
    # 1. 주가 및 이평선 차트 (ax1)
    ax1.plot(df.index, df['Close'], label='Close Price', color='black')
    
    if 'MA5' in df.columns: ax1.plot(df.index, df['MA5'], label='MA5', color='cyan', alpha=0.7)
    if 'MA20' in df.columns: ax1.plot(df.index, df['MA20'], label='MA20', color='green')
    if 'MA60' in df.columns: ax1.plot(df.index, df['MA60'], label='MA60', color='orange')
    if 'MA120' in df.columns: ax1.plot(df.index, df['MA120'], label='MA120', color='red', alpha=0.5)

    if 'BB_Upper' in df.columns:
        ax1.plot(df.index, df['BB_Upper'], 'g--', label='BB Upper', alpha=0.5)
        ax1.plot(df.index, df['BB_Lower'], 'r--', label='BB Lower', alpha=0.5)
        ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.05)
        
    ax1.set_title(f"{ticker} 분석 차트 (애널리스트 의견: {analyst_rec})", fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # 2. RSI/MFI 및 거래량 차트 (ax2)
    if show_momentum and num_subcharts > 1:
        current_ax = axes[1]
        show_mfi = 'E.' in strategy_type or ('MFI' in df.columns and ('RSI' not in df.columns or df['RSI'].isnull().all()))
        
        if show_mfi and 'MFI' in df.columns and not df['MFI'].isnull().all():
             current_ax.plot(df.index, df['MFI'], label='MFI (14)', color='brown')
             current_ax.axhline(80, color='red', linestyle='--', label='MFI 80 (Overbought)')
             current_ax.axhline(50, color='blue', linestyle=':', label='MFI 50')
             current_ax.axhline(20, color='green', linestyle='--', label='MFI 20 (Oversold)')
             current_ax.set_title("MFI Indicator")
        elif 'RSI' in df.columns and not df['RSI'].isnull().all():
             current_ax.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
             current_ax.axhline(70, color='red', linestyle='--', label='RSI 70 (Overbought)')
             current_ax.axhline(50, color='blue', linestyle=':', label='RSI 50')
             current_ax.axhline(30, color='green', linestyle='--', label='RSI 30 (Oversold)')
             current_ax.set_title("RSI Indicator")
        else:
            current_ax.set_title("Volume Chart")

        if 'Volume' in df.columns:
            ax2_vol = current_ax.twinx()
            ax2_vol.bar(df.index, df['Volume'], color='gray', alpha=0.3, label='Volume')
            ax2_vol.set_ylabel('Volume', color='gray')
            ax2_vol.tick_params(axis='y', labelcolor='gray')
        
        current_ax.legend(loc='upper left')
        current_ax.grid(True, alpha=0.3)
    
    # 3. MACD 차트 (ax3)
    if has_macd and num_subcharts > 2:
        ax3 = axes[-1]
        ax3.plot(df.index, df['MACD'], label='MACD Line', color='blue')
        ax3.plot(df.index, df['MACD_Signal'], label='Signal Line', color='red')
        ax3.bar(df.index, df['MACD'] - df['MACD_Signal'], label='Histogram', color='gray', alpha=0.5)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title("MACD Indicator")
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)


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

def display_ticker_info(ticker, df_analyzed, analyst_rec):
    st.markdown(f"### {ticker} 상세 정보")
    st.markdown(f"**🗣️ 애널리스트 의견:** **{analyst_rec.upper()}**")
    
    # plot_chart 함수가 분석된 DF를 사용하도록 수정
    fig = plot_chart(ticker, df_analyzed, "개별 조회", analyst_rec) 
    
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
    
    all_strategies = [
        "A. 강력 수급 폭발 (거래량 1.5배)", 
        "B. 단기/중기 이동평균선 골든크로스 (MA20 > MA60)", 
        "C. RSI 과매도 반등 (30 이하)", 
        "D. MACD 시그널선 상향 돌파", 
        "E. MFI 과매도 반등 (20 이하)", 
        "F. 볼린저밴드 상단 돌파", 
        "G. 장대양봉 및 짧은 꼬리", 
        "H. RSI 상승 다이버전스",         
        "I. MACD 상승 다이버전스",       
        "J. MA 이격도 과매도",            
    ]
    
    selected_strategies = st.sidebar.multiselect("원하는 타점을 모두 선택하세요 (OR 조건)", all_strategies)

    # --- 3️⃣ 스캔할 종목 목록 (총 100개 원상 복구) ---
    st.sidebar.header("3️⃣ 스캔할 종목 목록 (총 100개)")
    
    # 🌟 원상 복구된 100개 종목 리스트
    kosdaq_top50 = "000210.KQ, 000660.KQ, 000880.KQ, 001120.KQ, 001390.KQ, 001550.KQ, 002170.KQ, 002200.KQ, 002270.KQ, 002320.KQ, 002360.KQ, 002390.KQ, 003380.KQ, 003550.KQ, 003560.KQ, 003620.KQ, 003650.KQ, 004140.KQ, 004720.KQ, 004830.KQ, 005180.KQ, 005880.KQ, 005930.KQ, 006400.KQ, 007680.KQ, 008770.KQ, 009190.KQ, 010060.KQ, 010120.KQ, 010140.KQ, 011070.KQ, 012280.KQ, 012450.KQ, 012750.KQ, 013420.KQ, 013640.KQ, 013700.KQ, 014990.KQ, 015350.KQ, 015760.KQ, 016600.KQ, 018000.KQ, 018260.KQ, 019550.KQ, 020660.KQ, 023590.KQ, 024740.KQ, 025680.KQ, 028080.KQ, 028300.KQ"
    kospi_low50 = "000100.KS, 000180.KS, 000210.KS, 000220.KS, 000230.KS, 000300.KS, 000320.KS, 000370.KS, 000480.KS, 000500.KS, 000520.KS, 000540.KS, 000650.KS, 000670.KS, 000810.KS, 000860.KS, 000880.KS, 000950.KS, 000970.KS, 001040.KS, 001060.KS, 001070.KS, 001080.KS, 001120.KS, 001140.KS, 001210.KS, 001230.KS, 001250.KS, 001270.KS, 001380.KS, 001390.KS, 001430.KS, 001520.KS, 001550.KS, 001570.KS, 001630.KS, 001740.KS, 001780.KS, 001800.KS, 001820.KS, 001940.KS, 001950.KS, 002020.KS, 002030.KS, 002070.KS, 002170.KS, 002200.KS, 002210.KS, 002240.KS, 002270.KS"

    default_tickers = kospi_low50 + ", " + kosdaq_top50
    st.sidebar.markdown("현재 **코스피 소형주 50개 + 코스닥 대형주 50개 (총 100개)**가 자동 설정되었습니다. **(수정 가능)**")
    tickers_input = st.sidebar.text_area("티커 목록 (쉼표 구분)", default_tickers, height=200) 
    
    # --- 4️⃣ 텔레그램 알림 설정 (고정 및 자동 활성화 유지) ---
    st.sidebar.header("4️⃣ 텔레그램 알림 설정 (자동)")
    tg_token = "7983927652:AAH8RRQpyJaika94NVmbmowvDIu5wHgfyWo"
    tg_chat_id = "1786596437"
    enable_alert = True 
    st.sidebar.success("✅ 텔레그램 알림이 코드로 고정/활성화되었습니다.")
    st.sidebar.markdown(f"**챗 ID:** `{tg_chat_id}` (이 ID로봇이 메시지를 보낼 수 있도록 **반드시** 이 ID를 가진 채팅방에 봇을 추가해주세요.)")
    
    # --- 메인 화면 로직 ---
    
    if st.button("🔍 타점 전략 스캔 시작"):
        if not selected_strategies:
            st.warning("분석할 전략을 1개 이상 선택해주세요. 🧘")
            return

        tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
        
        st.write(f"### 🕵️ '{', '.join(selected_strategies)}' 전략으로 총 {len(tickers)}개 종목을 스캔합니다...")
        
        found_count = 0
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            
            # 1. 정보 가져오기 (마켓캡, 애널리스트 의견 등)
            info, market_cap_usd, analyst_rec = get_stock_info(ticker)
            
            # 2. 분석 실행 (matched_reasons와 df_analyzed 반환)
            matched_reasons, df_analyzed = analyze_stock(ticker, selected_strategies)
            
            # --- 결과 처리 ---
            if matched_reasons:
                found_count += 1
                
                # 3. 매칭된 경우, 가격 정보 추출
                try:
                    if df_analyzed.empty or len(df_analyzed) < 2:
                        raise ValueError("데이터 분석 결과가 불충분합니다.")
                        
                    today_data = df_analyzed.iloc[-1]
                    yesterday_data = df_analyzed.iloc[-2]
                    
                    current_price = today_data['Close']
                    change_pct = ((today_data['Close'] - yesterday_data['Close']) / yesterday_data['Close']) * 100
                    
                    # Streamlit UI에 결과 표시
                    st.markdown(f"#### 🎯 {ticker} ({info.get('shortName', 'N/A')}) - 타점 발견!")
                    st.markdown(f"**📈 현재가:** {current_price:,.2f} | **변동률:** <span style='color:{'red' if change_pct >= 0 else 'blue'}'>{change_pct:+.2f}%</span>", unsafe_allow_html=True)
                    st.markdown(f"**💰 시가총액:** {market_cap_usd:.2f} 억 달러")
                    
                    # 차트 생성 및 표시
                    fig = plot_chart(ticker, df_analyzed, 
                                     ", ".join([m['strategy'] for m in matched_reasons]), 
                                     analyst_rec)
                    if fig:
                        st.pyplot(fig)
                    
                    # Streamlit 리스트
                    st.markdown("---")
                    st.markdown("**📌 발견된 전략:**")
                    for reason_data in matched_reasons:
                        st.markdown(f"- **{reason_data['strategy']}**: {reason_data['reason']}")
                    st.markdown("---")

                    # 4. 텔레그램 알림 전송 (개선된 메시지)
                    if enable_alert:
                        # 🌟 개선된 텔레그램 메시지 포맷
                        header = f"<b>🚨 타점 포착! {ticker} ({info.get('shortName', 'N/A')})</b>"
                        price_color = "red" if change_pct >= 0 else "blue"
                        price_line = f"현재가: <b>{current_price:,.2f}</b> | 변동률: <b style='color:{price_color}'>{change_pct:+.2f}%</b>"
                        
                        strategy_lines = []
                        for reason_data in matched_reasons:
                            strategy_lines.append(f"• <b>{reason_data['strategy']}</b>\n  └ {reason_data['reason']}")
                        
                        telegram_message = f"{header}\n\n{price_line}\n\n<u>포착 전략 ({len(matched_reasons)}개)</u>\n" + "\n".join(strategy_lines)

                        send_telegram_msg(tg_token, tg_chat_id, telegram_message)
                
                except Exception as e:
                    st.error(f"🚨 {ticker} 데이터 처리 중 오류 발생 (차트/알림 건너뜀): {e}")
                    print(f"[{ticker}] 오류 상세: {e}")
            
            # 프로그레스 바 업데이트
            progress_bar.progress((i + 1) / len(tickers))

        progress_bar.empty()
        st.success(f"✅ 스캔 완료! 총 {len(tickers)}개 종목 중 {found_count}개 종목에서 타점을 발견했습니다.")

    # --- 개별 종목 분석 섹션 ---
    if st.sidebar.button("📊 개별 종목 조회") and single_ticker:
        st.sidebar.markdown("---")
        st.sidebar.header("개별 조회 결과")
        
        info, market_cap_usd, analyst_rec = get_stock_info(single_ticker)
        
        # 데이터 분석 실행 (df_analyzed를 얻음)
        matched_reasons, df_analyzed = analyze_stock(single_ticker, all_strategies)

        if df_analyzed is not None and not df_analyzed.empty and 'Close' in df_analyzed.columns:
            display_ticker_info(single_ticker, df_analyzed, analyst_rec)
            
            # 개별 조회 시에도 전략 검사 수행 (전체 전략 기준)
            if matched_reasons:
                 st.markdown("#### ✨ 현재 전략 일치 여부 (전체 전략 기준):")
                 for reason_data in matched_reasons:
                    st.markdown(f"- **{reason_data['strategy']}**: {reason_data['reason']}")
            else:
                st.markdown("#### ✨ 현재 일치하는 전략 타점이 없습니다. (전체 전략 기준)")
        else:
            st.sidebar.error(f"티커 **{single_ticker}**의 데이터를 불러올 수 없거나 유효하지 않습니다.")

# 앱 실행
if __name__ == '__main__':
    main()
