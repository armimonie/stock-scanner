import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
import time

# --- 텔레그램 알림 함수 (HTML 포맷 지정 추가) ---
def send_telegram_msg(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return
    try:
        url = f"https://api.telegram.com/bot{bot_token}/sendMessage"
        params = {
            'chat_id': chat_id, 
            'text': message,
            'parse_mode': 'HTML'  # HTML 태그를 사용하도록 설정
        }
        # API 호출 시 지연 시간 추가 (너무 빠른 요청 방지)
        time.sleep(0.5) 
        requests.get(url, params=params)
    except Exception as e:
        # st.error 대신 console log로 처리하여 앱 UI를 방해하지 않음
        print(f"텔레그램 전송 실패: {e}")

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
        if 'MA20' in df_copy.columns and not df_copy['MA20'].isnull().all():
            # 이격도 = (현재 종가 / MA20) * 100
            df_copy['Disparity'] = (df_copy['Close'] / df_copy['MA20']) * 100
        else:
            df_copy['Disparity'] = np.nan
        
    except Exception as e:
        # 지표 계산 중 오류 발생 시 None 반환
        print(f"지표 계산 오류 발생: {e}")
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
    except Exception as e:
        print(f"티커 {ticker} 데이터 로드 실패: {e}")
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
    # 오늘 포함 최근 n+1일 데이터 (오늘을 포함해야 today가 됨)
    recent_df_full = df.iloc[-(n+1):] 
    
    if len(recent_df_full) >= 2:
        # 오늘 날짜
        today_data = recent_df_full.iloc[-1]
        # 오늘을 제외한 이전 n일 데이터
        recent_df = recent_df_full.iloc[:-1] 
    else:
        return matched_reasons # 데이터 부족

    # V6.2: 다이버전스 전제 조건: 주가는 n일 동안 저점을 갱신했는가?
    price_low_new = today_data.get('Close', np.nan) 
    price_low_old = recent_df['Close'].min()
    
    # 주가 하락 (새 저점 < 이전 n일간 저점)이 전제되어야 상승 다이버전스 검색 가능
    is_price_diverging = not pd.isna(price_low_new) and not pd.isna(price_low_old) and price_low_new < price_low_old 
    
    # 전략 H: RSI 상승 다이버전스 (RSI 저점 상승)
    if "H. RSI 상승 다이버전스" in selected_strategies and is_price_diverging and 'RSI' in df.columns:
        rsi_low_new = today_data.get('RSI', np.nan)
        rsi_low_old = recent_df['RSI'].min() if 'RSI' in recent_df.columns else np.nan
        
        # RSI 저점 상승 (새 저점 > 이전 저점) and RSI 40 이하에서 발생 (신뢰도 높음)
        if not pd.isna(rsi_low_new) and not pd.isna(rsi_low_old) and rsi_low_new > rsi_low_old and rsi_low_new < 40: 
            matched_reasons.append({"strategy": "H. RSI 상승 다이버전스", "reason": f"⚡️ 주가 저점 하락에도 RSI({rsi_low_new:.1f})는 상승하여 **강력한 추세 반전(다이버전스)** 신호 포착."})

    # 전략 I: MACD 상승 다이버전스 (MACD 저점 상승)
    if "I. MACD 상승 다이버전스" in selected_strategies and is_price_diverging and 'MACD' in df.columns:
        macd_low_new = today_data.get('MACD', np.nan)
        macd_low_old = recent_df['MACD'].min() if 'MACD' in recent_df.columns else np.nan
        
        # MACD 저점 상승 (새 저점 > 이전 저점) and MACD 0선 이하에서 발생 (신뢰도 높음)
        if not pd.isna(macd_low_new) and not pd.isna(macd_low_old) and macd_low_new > macd_low_old and today_data.get('MACD', 1) < 0:
            matched_reasons.append({"strategy": "I. MACD 상승 다이버전스", "reason": f"✨ 주가 하락에도 MACD({macd_low_new:.2f})는 상승하여 **중기 추세 반전(다이버전스)** 신호 포착."})

    # 전략 J: MA 이격도 과매도 (20일선 대비 95% 이하)
    if "J. MA 이격도 과매도" in selected_strategies and 'Disparity' in df.columns:
        tdisparity = today_data.get('Disparity', np.nan)
        # 이격도가 95% 이하: 주가가 20일 이동평균선보다 5% 이상 하락
        if not pd.isna(tdisparity) and tdisparity <= 95.0:
            matched_reasons.append({"strategy": "J. MA 이격도 과매도", "reason": f"📉 이격도({tdisparity:.1f}%)가 95% 이하로 **단기 낙폭 과대** 상태입니다. 평균 회귀 기대."})
            
    return matched_reasons

# ---------------------------------------------------------
# 2. 차트 시각화 함수
# ---------------------------------------------------------
def plot_chart(ticker, df, strategy_type, analyst_rec):
    if df is None or df.empty or 'Close' not in df.columns:
        return None
        
    has_macd = 'MACD' in df.columns and not df['MACD'].isnull().all()
    
    # RSI/MFI가 모두 NaN인 경우를 대비하여 차트 개수 조정
    show_momentum = ('RSI' in df.columns and not df['RSI'].isnull().all()) or \
                    ('MFI' in df.columns and not df['MFI'].isnull().all()) or \
                    ('Volume' in df.columns and not df['Volume'].isnull().all())
    
    num_subcharts = 1 # 기본 주가 차트
    if show_momentum:
        num_subcharts += 1
    if has_macd:
        num_subcharts += 1
        
    if num_subcharts == 1:
        # MACD, RSI/MFI, Volume 모두 없는 경우
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        axes = [ax1]
    elif num_subcharts == 2:
        # MACD만 없거나, RSI/MFI/Volume만 없는 경우 (후자는 거의 없겠지만)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax1, ax2 = axes
    else: # num_subcharts == 3
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [4, 1, 1]})
        ax1, ax2, ax3 = axes
    
    
    # 1. 주가 및 이평선 차트 (ax1)
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
        
    ax1.set_title(f"{ticker} 분석 차트 (애널리스트 의견: {analyst_rec})", fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # 2. RSI/MFI 및 거래량 차트 (ax2)
    if show_momentum:
        # ax2는 RSI/MFI 또는 Volume 차트
        current_ax = axes[1] if num_subcharts > 1 else ax1 # num_subcharts가 1이면 ax1에 겹쳐 그리는 것 방지

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
    if has_macd:
        ax3 = axes[-1] # 마지막 서브차트가 MACD
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

def display_ticker_info(ticker, df, analyst_rec):
    st.markdown(f"### {ticker} 상세 정보")
    st.markdown(f"**🗣️ 애널리스트 의견:** **{analyst_rec.upper()}**")
    
    # 지표 계산을 먼저 시도하여 차트 생성에 사용
    df_analyzed = calculate_indicators(df)
    
    # plot_chart 함수가 분석된 DF를 사용하도록 수정
    fig = plot_chart(ticker, df_analyzed if df_analyzed is not None else df, "개별 조회", analyst_rec) 
    
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
    # 예시: 삼성전자 005930.KS, 애플 AAPL
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
    
    # 코스닥 상위 50개 종목 리스트 (대형주 위주) - 예시 티커로 변경 (yfinance 호환성을 고려하여)
    # yfinance가 모든 코스닥 티커를 잘 지원하지 않으므로, 테스트용으로 제한된 목록 사용
    kosdaq_top50_example = "035720.KQ, 066970.KQ, 041190.KQ, 096610.KQ, 000210.KQ" # 카카오게임즈, 엘앤에프 등
    
    # 코스피 하위 50개 종목 리스트 (소형주 위주) - 예시 티커로 변경
    kospi_low50_example = "005930.KS, 005380.KS, 035420.KS, 000660.KS, 012330.KS" # 삼성전자, 현대차, 네이버, SK하이닉스, 현대모비스
    
    # 두 리스트를 합쳐서 기본값 설정 (테스트를 위해 수를 줄였습니다)
    default_tickers = kospi_low50_example + ", " + kosdaq_top50_example + ", TSLA, MSFT"
    st.sidebar.markdown("현재 **코스피/코스닥 대형주 + 해외주식 (총 10개)**가 자동 설정되었습니다. **(수정 가능)**")
    tickers_input = st.sidebar.text_area("티커 목록 (쉼표 구분)", default_tickers) 
    
    # --- 4️⃣ 텔레그램 알림 설정 (고정 및 자동 활성화 유지) ---
    st.sidebar.header("4️⃣ 텔레그램 알림 설정 (자동)")
    # 사용자님의 고정된 토큰 및 ID 사용
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

        tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
        
        st.write(f"### 🕵️ '{', '.join(selected_strategies)}' 전략으로 총 {len(tickers)}개 종목을 스캔합니다...")
        
        found_count = 0
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            
            # --- 정보 가져오기 ---
            info, market_cap_usd, analyst_rec = get_stock_info(ticker)
            
            # --- 분석 실행 ---
            matched_reasons = analyze_stock(ticker, selected_strategies)
            
            # --- 결과 처리 ---
            if matched_reasons:
                found_count += 1
                
                # Streamlit UI에 결과 표시
                st.markdown(f"#### 🎯 {ticker} ({info.get('shortName', 'N/A')}) - 타점 발견!")
                st.markdown(f"**💰 시가총액:** {market_cap_usd:.2f} 억 달러")
                
                # 차트 생성 및 표시를 위해 다시 데이터 로드 및 분석
                # analyze_stock에서 이미 데이터 로드 및 분석을 했지만, 
                # 차트 함수가 DF를 요구하므로 여기서 한 번 더 처리하거나 analyze_stock이 DF를 반환하도록 수정 필요.
                # 편의를 위해 여기서 다시 yf.Ticker를 호출합니다.
                df = yf.Ticker(ticker).history(period="1y")
                df_analyzed = calculate_indicators(df)
                
                fig = plot_chart(ticker, df_analyzed if df_analyzed is not None else df, 
                                 ", ".join([m['strategy'] for m in matched_reasons]), 
                                 analyst_rec)
                if fig:
                    st.pyplot(fig)
                
                st.markdown("**📌 발견된 전략:**")
                telegram_message_parts = [f"<b>{ticker}</b> ({info.get('shortName', 'N/A')}) 타점 발견!"]
                
                for reason_data in matched_reasons:
                    # HTML 포맷을 위해 <b> 태그 사용
                    st.markdown(f"- **{reason_data['strategy']}**: {reason_data['reason']}")
                    telegram_message_parts.append(f"- {reason_data['strategy']}: {reason_data['reason']}")
                
                st.markdown("---")

                # 텔레그램 알림 전송 (하나의 알림으로 통합)
                if enable_alert:
                    telegram_message = "\n".join(telegram_message_parts)
                    send_telegram_msg(tg_token, tg_chat_id, telegram_message)
            
            # 프로그레스 바 업데이트
            progress_bar.progress((i + 1) / len(tickers))

        progress_bar.empty()
        st.success(f"✅ 스캔 완료! 총 {len(tickers)}개 종목 중 {found_count}개 종목에서 타점을 발견했습니다.")

    # --- 개별 종목 분석 섹션 ---
    # 개별 종목 분석을 위한 로직 추가
    if st.sidebar.button("📊 개별 종목 조회") and single_ticker:
        st.sidebar.markdown("---")
        st.sidebar.header("개별 조회 결과")
        
        # 데이터 가져오기 및 분석
        ticker_obj = yf.Ticker(single_ticker)
        df = ticker_obj.history(period="1y") 
        
        info, market_cap_usd, analyst_rec = get_stock_info(single_ticker)

        if not df.empty and 'Close' in df.columns:
            display_ticker_info(single_ticker, df, analyst_rec)
            # 개별 조회 시에도 전략 검사 수행 (옵션)
            matched_reasons = analyze_stock(single_ticker, all_strategies)
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
    # 텔레그램 토큰이 노출되는 것을 방지하기 위해 실제 환경에서는 환경 변수를 사용하는 것이 좋습니다.
    # 이 코드는 Streamlit 환경에서 직접 실행될 때의 예시입니다.
    main()
