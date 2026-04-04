import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time
import numpy as np
import re

GMAIL_ADDRESS  = os.environ['GMAIL_ADDRESS']
GMAIL_APP_PASS = os.environ['GMAIL_APP_PASS']
SEND_TO        = os.environ['SEND_TO']

jst = pytz.timezone('Asia/Tokyo')
now = datetime.now(jst)
now_str = now.strftime('%Y-%m-%d %H:%M')
print(f"実行開始: {now_str}")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Referer": "https://jp.investing.com/"
}

# ========================================
# 日本10年債金利
# ========================================
jgb_val = None
jgb_change = "-"
jgb_pct = "-"
try:
    res_inv = requests.get(
        "https://jp.investing.com/rates-bonds/japan-10-year-bond-yield",
        headers=headers, timeout=8
    )
    soup_inv = BeautifulSoup(res_inv.text, 'html.parser')
    curr_tag   = soup_inv.find('div', {'data-test': 'instrument-price-last'})
    change_tag = soup_inv.find('span', {'data-test': 'instrument-price-change'})
    pct_tag    = soup_inv.find('span', {'data-test': 'instrument-price-change-percent'})
    if curr_tag:
        jgb_val = float(curr_tag.get_text(strip=True))
    if change_tag:
        chg = float(change_tag.get_text(strip=True))
        jgb_change = f"{chg:+.3f}"
    if pct_tag:
        jgb_pct = pct_tag.get_text(strip=True).strip('()')
except Exception as e:
    print(f"JGB取得エラー: {e}")

# ========================================
# 日経225構成銘柄を自動取得
# ========================================
def fetch_nikkei225_codes():
    try:
        url = "https://en.wikipedia.org/wiki/Nikkei_225"
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, 'html.parser')
        codes = []
        for table in soup.find_all('table', {'class': 'wikitable'}):
            for row in table.find_all('tr')[1:]:
                cells = row.find_all('td')
                for cell in cells:
                    text = cell.get_text(strip=True)
                    m = re.search(r'\b(\d{4})\b', text)
                    if m:
                        codes.append(m.group(1))
                        break
        codes = list(dict.fromkeys(codes))
        if len(codes) >= 200:
            print(f"Wikipedia取得成功：{len(codes)}銘柄")
            return codes
    except Exception as e:
        print(f"Wikipedia取得失敗: {e}")

    print("フォールバックリストを使用")
    return [
        "1332","1333","1605","1721","1801","1802","1803","1808","1812","1925",
        "1928","1963","2002","2269","2282","2413","2432","2501","2502","2503",
        "2531","2578","2579","2587","2593","2695","2702","2801","2802","2871",
        "2914","3086","3099","3105","3289","3382","3401","3402","3405","3407",
        "3436","3659","3861","3863","3893","3941","4004","4005","4021","4042",
        "4043","4061","4063","4151","4183","4188","4208","4324","4452","4502",
        "4503","4506","4507","4519","4523","4543","4568","4578","4631","4642",
        "4689","4704","4751","4755","4901","4902","4911","5001","5020","5101",
        "5105","5108","5110","5201","5202","5214","5232","5233","5301","5332",
        "5333","5401","5406","5411","5413","5423","5631","5703","5706","5707",
        "5711","5713","5714","5715","5726","5727","5801","5802","5803","5901",
        "6098","6103","6113","6146","6178","6273","6301","6302","6305","6326",
        "6361","6367","6368","6369","6370","6471","6472","6473","6501","6503",
        "6504","6506","6508","6586","6594","6645","6647","6674","6701","6702",
        "6703","6706","6724","6752","6753","6758","6762","6770","6841","6857",
        "6861","6902","6952","6954","6971","6976","6981","6988","7003","7004",
        "7011","7012","7013","7182","7186","7201","7202","7203","7205","7206",
        "7207","7211","7261","7267","7269","7270","7272","7731","7733","7735",
        "7741","7751","7752","7762","7832","7911","7912","7974","8001","8002",
        "8003","8015","8031","8035","8053","8058","8233","8252","8267","8303",
        "8304","8306","8308","8309","8316","8411","8591","8601","8604","8630",
        "8697","8725","8750","8766","8795","9001","9005","9007","9008","9009",
        "9020","9021","9022","9064","9101","9104","9107","9202","9301","9412",
        "9432","9433","9434","9501","9502","9503","9531","9532","9602","9613",
        "9616","9735","9766","9983","9984"
    ]

nikkei225_codes = fetch_nikkei225_codes()

# ========================================
# 決算日フラグ取得
# ========================================
def get_earnings_flag(code):
    try:
        ticker = yf.Ticker(f"{code}.T")
        cal = ticker.calendar
        if cal is None: return ""
        earnings_date = None
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            earnings_date = cal['Earnings Date'][0] if isinstance(cal['Earnings Date'], list) else cal['Earnings Date']
        elif not isinstance(cal, dict) and ('Earnings Date' in cal.columns or 'Earnings Date' in cal.index):
            earnings_date = cal.loc['Earnings Date'].iloc[0] if 'Earnings Date' in cal.index else cal['Earnings Date'].iloc[0]
        if earnings_date is None: return ""
        if hasattr(earnings_date, 'tzinfo') and earnings_date.tzinfo is not None:
            earnings_date = earnings_date.replace(tzinfo=None)
        diff = (earnings_date - datetime.now().replace(tzinfo=None)).days
        if 0 <= diff <= 7:
            return f"決算{earnings_date.month}/{earnings_date.day}"
        return ""
    except:
        return ""

# ========================================
# 日本語名の取得
# ========================================
def get_japanese_name(code):
    try:
        url = f"https://finance.yahoo.co.jp/quote/{code}.T"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
        soup = BeautifulSoup(res.text, 'html.parser')
        name_tag = soup.select_one('header h1')
        if name_tag:
            name = name_tag.text
            name = re.sub(r'の株価.*', '', name)
            name = re.sub(r'\(株\)|（株）', '', name)
            name = name.strip()
            return name[:12]
    except:
        pass
    try:
        info = yf.Ticker(f"{code}.T").info
        for key in ('displayName', 'longName', 'shortName'):
            val = info.get(key)
            if val: return val[:12]
    except:
        pass
    return code

# ========================================
# 信用倍率取得（Yahoo Finance Japan）
# ========================================
def get_shinyo_bairitu(code):
    """Yahoo Finance Japanから信用倍率を取得。取得失敗時はNoneを返す。
    500エラー時は3s待ってリトライ（最大2回）。
    """
    url = f"https://finance.yahoo.co.jp/quote/{code}.T"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
               "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"}
    for attempt in range(3):
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 500:
                if attempt < 2:
                    wait = 10 * (attempt + 1)  # 1回目:10s, 2回目:20s
                    time.sleep(wait)
                    continue
                return None
            if res.status_code != 200:
                return None
            soup = BeautifulSoup(res.text, 'html.parser')
            for dt in soup.find_all('dt'):
                if '信用倍率' in dt.get_text(strip=True):
                    dd = dt.find_next_sibling('dd')
                    if dd:
                        m = re.search(r'[\d,]+\.?\d*', dd.get_text(strip=True))
                        if m:
                            return float(m.group().replace(',', ''))
            return None
        except Exception as e:
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
            else:
                print(f"信用倍率取得エラー {code}: {e}")
    return None

# ========================================
# STEP 1：売買代金上位50銘柄を選出
# ========================================
print("STEP 1：売買代金上位50銘柄を取得中...")

volume_scores = []
for code in nikkei225_codes:
    try:
        df = yf.Ticker(f"{code}.T").history(period="5d")
        if df.empty or len(df) < 1: continue
        latest = df.iloc[-1]
        trading_value = latest['Volume'] * latest['Close'] / 1e8
        volume_scores.append({'code': code, 'trading_value': trading_value})
    except:
        continue
    time.sleep(0.05)

volume_scores = sorted(volume_scores, key=lambda x: x['trading_value'], reverse=True)[:50]

top50_stocks = []
for v in volume_scores:
    name = get_japanese_name(v['code'])
    top50_stocks.append((v['code'], name))
    time.sleep(0.1)

print(f"上位50銘柄選出完了")

# ========================================
# HVN計算（改善①：ビン20・直近1ヶ月×2重み）
# ========================================
def calc_hvn(df, curr, atr=None, atr_mult=2.0, fallback_mult=3.0):
    """
    ビン数20に増加、直近1ヶ月データに×2の重みをつけてHVNを算出。
    atr指定時: curr±ATR×atr_mult 以上離れたHVNを選択。
              該当HVNなし→ curr±ATR×fallback_mult をフォールバック目標とする。
    戻り値: (sup_price, res_price)
    """
    try:
        cutoff = df['Date'].max() - pd.Timedelta(days=30)
        weights = np.where(df['Date'] >= cutoff, 2.0, 1.0)
        weighted_volume = df['Volume'] * weights

        bin_result = pd.cut(df['Close'], bins=20, labels=False, retbins=True)
        pbins      = bin_result[0]
        bin_edges  = bin_result[1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        vol_by_bin = pd.Series(weighted_volume.values, index=df.index).groupby(pbins).sum()
        top5       = vol_by_bin.nlargest(5).index
        hvn        = sorted([bin_centers[i] for i in top5 if i < len(bin_centers)])

        if atr and atr > 0:
            min_dist = atr * atr_mult
            sup = max([p for p in hvn if p <= curr - min_dist], default=None)
            res = min([p for p in hvn if p >= curr + min_dist], default=None)
            if sup is None: sup = curr - atr * fallback_mult
            if res is None: res = curr + atr * fallback_mult
        else:
            sup = max([p for p in hvn if p < curr], default=None)
            res = min([p for p in hvn if p > curr], default=None)
        return sup, res
    except:
        return None, None

# ========================================
# MACDクロス鮮度チェック（改善④）
# ========================================
def calc_macd_cross(df, direction):
    """
    直近3日以内にMACDがシグナルをクロスし、かつ推奨方向と一致する場合10点。
    direction: "買い" or "売り"
    """
    try:
        macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd_df is None or macd_df.empty: return 0
        # カラム名を動的取得
        macd_col   = [c for c in macd_df.columns if 'MACD_' in c and 'MACDs' not in c and 'MACDh' not in c][0]
        signal_col = [c for c in macd_df.columns if 'MACDs_' in c][0]

        macd_line   = macd_df[macd_col].dropna()
        signal_line = macd_df[signal_col].dropna()
        if len(macd_line) < 4: return 0

        # 直近4本でクロス検出（=直近3日以内）
        for i in range(-3, 0):
            prev_diff = macd_line.iloc[i-1] - signal_line.iloc[i-1]
            curr_diff = macd_line.iloc[i]   - signal_line.iloc[i]
            golden = prev_diff < 0 and curr_diff >= 0  # 買いクロス
            dead   = prev_diff > 0 and curr_diff <= 0  # 売りクロス

            if direction == "買い" and golden: return 10
            if direction == "売り" and dead:   return 10
        return 0
    except:
        return 0

# ========================================
# ジグザグ + MA75 トレンド判定
# ========================================
def calc_zigzag_trend(df, atr_val, ma75_col='MA75'):
    """
    直近20本のジグザグ分析でトレンドを判定。
    転換点基準: ATR×1.5以上の値動きで高値・安値が更新された時点。
    戻り値: (zz_trend, ma75_up)
      zz_trend : '上昇' / '下降' / 'ボックス'
      ma75_up  : True(上向き) / False(下向き) / None(判定不能)
    """
    recent = df.tail(21).dropna(subset=['High', 'Low', 'Close']).tail(20).copy().reset_index(drop=True)
    threshold = atr_val * 1.0
    if not np.isfinite(threshold) or threshold <= 0 or len(recent) < 5:
        raise ValueError("threshold or data insufficient")

    # ジグザグ転換点を検出
    direction    = None          # 'up' or 'down'
    ref_high     = recent['High'].iloc[0]
    ref_low      = recent['Low'].iloc[0]
    ref_high_idx = 0
    ref_low_idx  = 0
    pivots = []  # (idx, price, 'H' or 'L')

    for i in range(1, len(recent)):
        h = recent['High'].iloc[i]
        l = recent['Low'].iloc[i]

        if direction is None:
            if h > ref_high: ref_high = h; ref_high_idx = i
            if l < ref_low:  ref_low  = l; ref_low_idx  = i
            if ref_high - ref_low >= threshold:
                if ref_high_idx < ref_low_idx:
                    pivots.append((ref_high_idx, ref_high, 'H'))
                    direction = 'down'
                    ref_low = l; ref_low_idx = i
                else:
                    pivots.append((ref_low_idx, ref_low, 'L'))
                    direction = 'up'
                    ref_high = h; ref_high_idx = i
        elif direction == 'up':
            if h > ref_high: ref_high = h; ref_high_idx = i
            if ref_high - l >= threshold:
                pivots.append((ref_high_idx, ref_high, 'H'))
                direction = 'down'
                ref_low = l; ref_low_idx = i
        else:  # down
            if l < ref_low: ref_low = l; ref_low_idx = i
            if h - ref_low >= threshold:
                pivots.append((ref_low_idx, ref_low, 'L'))
                direction = 'up'
                ref_high = h; ref_high_idx = i

    highs = [p[1] for p in pivots if p[2] == 'H']
    lows  = [p[1] for p in pivots if p[2] == 'L']

    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1] > highs[-2]
        hl = lows[-1]  > lows[-2]
        lh = highs[-1] < highs[-2]
        ll = lows[-1]  < lows[-2]
        if hh and hl:   zz_trend = '上昇'
        elif lh and ll: zz_trend = '下降'
        else:           zz_trend = 'ボックス'
    elif len(highs) >= 2:
        zz_trend = '下降' if highs[-1] < highs[-2] else 'ボックス'
    elif len(lows) >= 2:
        zz_trend = '上昇' if lows[-1] > lows[-2] else 'ボックス'
    else:
        zz_trend = 'ボックス'

    # MA75 方向判定（直近6本で5日分の傾き）
    ma75_up = None
    if ma75_col in df.columns:
        ma75_vals = df[ma75_col].dropna().tail(6)
        if len(ma75_vals) >= 6:
            ma75_up = bool(ma75_vals.iloc[-1] > ma75_vals.iloc[0])

    return zz_trend, ma75_up

# ========================================
# STEP 2：データ取得・raw値収集
# ========================================
def calc_raw(code, name):
    try:
        df = yf.Ticker(f"{code}.T").history(period="6mo", auto_adjust=False)
        if df.empty or len(df) < 20: return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.reset_index()

        df['RSI']  = ta.rsi(df['Close'], length=14)
        df['MA25'] = df['Close'].rolling(25).mean()
        df['MA75'] = df['Close'].rolling(75).mean()
        atr_series = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        recent = df.tail(20)
        latest = df.iloc[-1]
        curr   = latest['Close']
        rsi    = latest['RSI'] if not pd.isna(latest['RSI']) else 50

        change_pct = ((curr - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100 if len(df) >= 2 else 0.0)

        ma25 = latest['MA25']
        ma75 = latest['MA75']
        ma_divergence = (ma25 - ma75) / ma75 * 100 if not pd.isna(ma25) and not pd.isna(ma75) and ma75 != 0 else 0.0

        # トレンド方向（表示用）
        trend = '上昇' if ma_divergence > 0 else '下降'

        atr_val = atr_series.dropna().iloc[-1] if atr_series is not None and not atr_series.isna().all() else 0

        # ========================================
        # デイトレ：ATRベース損切・利確（改善②）
        # ========================================
        dt_direction = "買い" if change_pct >= 0 else "売り"
        if dt_direction == "買い":
            dt_sup = curr - atr_val        # 損切り = 現在値 - 1ATR
            dt_res = curr + 2 * atr_val   # 利確   = 現在値 + 2ATR
        else:
            dt_sup = curr + atr_val        # 損切り = 現在値 + 1ATR
            dt_res = curr - 2 * atr_val   # 利確   = 現在値 - 2ATR
        dt_rr = 2.0 if atr_val > 0 else 0.0
        dt_rr_valid = atr_val > 0

        # ========================================
        # スイング用HVN（改善①）
        # ========================================
        # 利確目標：ATR×2以上離れた遠いHVN（なければATR×3フォールバック）
        sup_tp_p, res_tp_p = calc_hvn(df, curr, atr=atr_val)
        # 損切ライン：現在値に最も近いHVN（なければATR×1フォールバック）
        sup_sl_p, res_sl_p = calc_hvn(df, curr)
        buy_sl  = sup_sl_p if sup_sl_p is not None else (curr - atr_val if atr_val > 0 else None)
        sell_sl = res_sl_p if res_sl_p is not None else (curr + atr_val if atr_val > 0 else None)

        # RR計算：利確=遠HVN、損切=近HVN
        rr_buy = rr_sell = 0.0
        if res_tp_p is not None and buy_sl is not None and curr != buy_sl:
            rr_buy  = abs(res_tp_p - curr) / abs(curr - buy_sl)
        if sup_tp_p is not None and sell_sl is not None and curr != sell_sl:
            rr_sell = abs(curr - sup_tp_p) / abs(sell_sl - curr)

        # 後段で使う変数名を従来名に合わせる
        sup_buy_p, res_buy_p = sup_tp_p, res_tp_p

        # ========================================
        # ボリバン %B（改善④）
        # ========================================
        pct_b = None
        try:
            bb_cols = ta.bbands(df['Close'], length=20, std=2)
            if bb_cols is not None:
                lower_col = [c for c in bb_cols.columns if 'BBL' in c][0]
                upper_col = [c for c in bb_cols.columns if 'BBU' in c][0]
                lower = bb_cols[lower_col].iloc[-1]
                upper = bb_cols[upper_col].iloc[-1]
                if upper != lower:
                    pct_b = (curr - lower) / (upper - lower)
        except:
            pass

        # ========================================
        # デイトレスコア
        # ========================================
        atr_pct   = atr_val / curr * 100
        avg_range = ((recent['High'] - recent['Low']) / recent['Close'] * 100).mean()
        trade_val = recent['Volume'].mean() * curr / 1e8
        price_s   = 1.0 if 500 <= curr <= 15000 else 0.3
        vol_cv    = recent['Volume'].std() / (recent['Volume'].mean() + 1e-10)

        daytrade_score = (
            min(atr_pct / 5 * 30, 30) +
            min(avg_range / 5 * 25, 25) +
            min(trade_val / 500 * 25, 25) +
            max(10 - vol_cv * 5, 0) +
            price_s * 10
        )

        # RSIスコア（35点・5段階テーブル）
        if 40 <= rsi <= 60:
            buy_rsi_score = 35; sell_rsi_score = 35
        elif 30 <= rsi < 40:
            buy_rsi_score = 28; sell_rsi_score = 14
        elif 60 < rsi <= 70:
            buy_rsi_score = 14; sell_rsi_score = 28
        elif rsi < 30:
            buy_rsi_score = 35; sell_rsi_score = 0
        else:  # rsi > 70
            buy_rsi_score = 0;  sell_rsi_score = 35

        return {
            'code':             code,
            'name':             name,
            'price':            curr,
            'change_pct':       change_pct,
            'rsi':              rsi,
            'trend':            trend,
            'ma_divergence':    ma_divergence,
            # デイトレ
            'dt_direction':   dt_direction,
            'dt_sup':         dt_sup,
            'dt_res':         dt_res,
            'dt_rr':          dt_rr,
            'dt_rr_valid':    dt_rr_valid,
            'atr_val':        atr_val,
            'daytrade_score': round(daytrade_score, 1),
            # スイング用raw
            'rr_buy_raw':     rr_buy,
            'rr_sell_raw':    rr_sell,
            'buy_rsi_score':  buy_rsi_score,
            'sell_rsi_score': sell_rsi_score,
            'sup_buy_price':  sup_buy_p,   # 買い利確（遠HVN上値）
            'res_buy_price':  res_buy_p,   # 売り利確（遠HVN下値）
            'buy_sl_price':   buy_sl,      # 買い損切（近HVN下値）
            'sell_sl_price':  sell_sl,     # 売り損切（近HVN上値）
            'pct_b':          pct_b,
            # MACDは方向確定後に計算するため一時保存用
            '_df':            df,
        }
    except Exception as e:
        print(f"エラー {code}: {e}")
        return None

print("STEP 2：全銘柄rawデータ収集中...")
all_results = []
for code, name in top50_stocks:
    r = calc_raw(code, name)
    if r: all_results.append(r)
    time.sleep(0.1)

# ========================================
# 相対評価でスイングスコアを確定（改善④）
# ========================================
def rank_score(values, higher_is_better=True, max_pts=30):
    n = len(values)
    if n <= 1: return [max_pts] * n
    arr = np.array(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and arr[order[j]] == arr[order[i]]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    if higher_is_better:
        return [(r - 1) / (n - 1) * max_pts for r in ranks]
    else:
        return [(n - r) / (n - 1) * max_pts for r in ranks]

def rr_score_abs(rr):
    """RRを絶対評価でスコア化（15点満点）"""
    if rr is None or rr <= 0:  return 0
    if rr >= 3.0: return 15
    if rr >= 2.0: return 12
    if rr >= 1.5: return 9
    if rr >= 1.0: return 6
    return 3

# トレンドスコア（20点・絶対評価）
# MA25 > MA75 → 買い20点/売り0点　MA25 ≤ MA75 → 買い0点/売り20点

for i, r in enumerate(all_results):
    # トレンドスコア（絶対評価）
    if r['ma_divergence'] > 0:
        buy_trend_sc_i  = 20
        sell_trend_sc_i = 0
    else:
        buy_trend_sc_i  = 0
        sell_trend_sc_i = 20

    # %Bスコア（20点・絶対評価）
    pct_b = r['pct_b']
    if pct_b is not None:
        buy_bb_score  = max(0, round((1 - pct_b) * 20, 1))   # 低いほど高得点
        sell_bb_score = max(0, round(pct_b * 20, 1))          # 高いほど高得点
    else:
        buy_bb_score = sell_bb_score = 0

    # 方向仮決定（MACDクロス計算のため）
    buy_score_tmp  = buy_trend_sc_i  + r['buy_rsi_score']  + rr_score_abs(r['rr_buy_raw'])  + buy_bb_score
    sell_score_tmp = sell_trend_sc_i + r['sell_rsi_score'] + rr_score_abs(r['rr_sell_raw']) + sell_bb_score
    tentative_dir  = "買い" if buy_score_tmp >= sell_score_tmp else "売り"

    # MACDクロス（10点・方向確定後）
    df_tmp     = r.pop('_df')  # DataFrameをpopして軽量化
    macd_score = calc_macd_cross(df_tmp, tentative_dir)

    buy_score  = buy_score_tmp  + (macd_score if tentative_dir == "買い" else 0)
    sell_score = sell_score_tmp + (macd_score if tentative_dir == "売り" else 0)

    buy_rr_valid  = (r['res_buy_price'] is not None and r['rr_buy_raw']  > 0)
    sell_rr_valid = (r['sup_buy_price'] is not None and r['rr_sell_raw'] > 0)

    if buy_score >= sell_score:
        r['swing_direction'] = "買い"
        r['swing_score']     = round(buy_score, 1)
        r['swing_sup']       = r['buy_sl_price']    # 損切 = 近HVN下値
        r['swing_res']       = r['res_buy_price']   # 利確 = 遠HVN上値
        r['swing_rr']        = r['rr_buy_raw']
        r['swing_rr_valid']  = buy_rr_valid
    else:
        r['swing_direction'] = "売り"
        r['swing_score']     = round(sell_score, 1)
        r['swing_sup']       = r['sell_sl_price']   # 損切 = 近HVN上値
        r['swing_res']       = r['sup_buy_price']   # 利確 = 遠HVN下値
        r['swing_rr']        = r['rr_sell_raw']
        r['swing_rr_valid']  = sell_rr_valid

    r['macd_score']     = macd_score
    r['buy_bb_score']   = buy_bb_score
    r['sell_bb_score']  = sell_bb_score

print("STEP 3：決算日フラグ取得中...")
earnings_flags = {}
for code, name in top50_stocks:
    flag = get_earnings_flag(code)
    if flag:
        earnings_flags[code] = flag
    time.sleep(0.1)

print("STEP 4：信用倍率取得中（Yahoo Finance Japan）...")
shinyo_data = {}
for i, (code, name) in enumerate(top50_stocks):
    ratio = get_shinyo_bairitu(code)
    if ratio is not None:
        shinyo_data[code] = ratio
    time.sleep(0.5)
    if (i + 1) % 20 == 0:
        time.sleep(10)  # 20件ごとにバッチ休止

# 信用倍率によるスイングスコア補正・ratioをresultに格納
for r in all_results:
    ratio = shinyo_data.get(r['code'])
    r['shinyo_ratio'] = ratio
    if ratio is not None:
        if r['swing_direction'] == "買い" and ratio >= 5.0:
            r['swing_score'] = round(r['swing_score'] - 10, 1)
        elif r['swing_direction'] == "売り" and ratio <= 0.5:
            r['swing_score'] = round(r['swing_score'] - 10, 1)

def get_nikkei_trend(fast=10, slow=25):
    """日経平均MA10/MA25でトレンド判定。返値: '買い'/'売り'/None"""
    try:
        df = yf.Ticker("^N225").history(period="3mo", auto_adjust=False)
        if df.empty or len(df) < slow:
            return None
        closes = df['Close'].values.astype(float)
        return '買い' if closes[-fast:].mean() >= closes[-slow:].mean() else '売り'
    except:
        return None

# ========================================
# TOP10選出
# ========================================
dt_top10    = sorted(all_results, key=lambda x: x['daytrade_score'], reverse=True)[:10]
# ルール2: 前日比 ±3% 超は急騰・急落銘柄として除外（バックテスト最適閾値）
swing_valid = [r for r in all_results if r['swing_rr_valid'] and abs(r['change_pct']) <= 3.0]

# 市場環境フィルター：日経MA10/MA25でトレンド判定し逆方向シグナルを除外
nikkei_market_trend = get_nikkei_trend(fast=10, slow=25)
if nikkei_market_trend is not None:
    filtered = [r for r in swing_valid if r['swing_direction'] == nikkei_market_trend]
    swing_valid = filtered if len(filtered) >= 5 else swing_valid  # 候補が5件未満ならフィルター解除

swing_top10 = sorted(swing_valid, key=lambda x: x['swing_score'], reverse=True)[:10]

dt_dir_map    = {r['code']: r['dt_direction']    for r in dt_top10}
swing_dir_map = {r['code']: r['swing_direction'] for r in swing_top10}

MISMATCH_CODES = {
    code for code in set(dt_dir_map) & set(swing_dir_map)
    if dt_dir_map[code] != swing_dir_map[code]
}

# ========================================
# マーケットデータ
# ========================================
def get_yf(symbol, is_rate=False):
    try:
        df = yf.Ticker(symbol).history(period="5d")
        if df.empty or len(df) < 2: return None, "-", "-"
        curr = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        change = curr - prev
        pct = (change / prev) * 100
        return (curr, f"{change:+.3f}", f"{pct:+.2f}%") if is_rate else (curr, f"{change:+,.1f}", f"{pct:+.2f}%")
    except:
        return None, "-", "-"

dow,    dow_c,    dow_p    = get_yf("^DJI")
sp,     sp_c,     sp_p     = get_yf("^GSPC")
nasdaq, nasdaq_c, nasdaq_p = get_yf("^IXIC")
usdjpy, fx_c,     fx_p     = get_yf("JPY=X",  is_rate=True)
nikkei, nk_c,     nk_p     = get_yf("^N225")
growth, gr_c,     gr_p     = get_yf("2516.T")
nkfut,  nf_c,     nf_p     = get_yf("NIY=F")
vix,    vix_c,    vix_p    = get_yf("^VIX",   is_rate=True)
tnx,    tnx_c,    tnx_p    = get_yf("^TNX",   is_rate=True)
oil,    oil_c,    oil_p    = get_yf("CL=F")
gold,   gold_c,   gold_p   = get_yf("GC=F")

def fmt(val, is_rate=False):
    if val is None: return "取得失敗"
    return f"{val:.3f}" if is_rate else f"{val:,.1f}"

items = [
    ("ダウ平均",     fmt(dow),          dow_c,    dow_p,    False),
    ("S&P 500",      fmt(sp),           sp_c,     sp_p,     False),
    ("Nasdaq",       fmt(nasdaq),       nasdaq_c, nasdaq_p, False),
    ("ドル円",       fmt(usdjpy, True), fx_c,     fx_p,     True),
    ("日経平均(現物)",fmt(nikkei),      nk_c,     nk_p,     False),
    ("東証グロース", fmt(growth),       gr_c,     gr_p,     False),
    ("日経先物",     fmt(nkfut),        nf_c,     nf_p,     False),
    ("日本10年債金利",f"{jgb_val:.3f}" if jgb_val else "取得失敗", jgb_change, jgb_pct, True),
    ("VIX指数",      fmt(vix, True),    vix_c,    vix_p,    True),
    ("米10年債金利", fmt(tnx, True),    tnx_c,    tnx_p,    True),
    ("WTI原油",      fmt(oil),          oil_c,    oil_p,    False),
    ("金先物",       fmt(gold),         gold_c,   gold_p,   False),
]

rows = ""
for label, val, change, pct, is_rate in items:
    color = "#d32f2f" if str(pct).startswith('+') else "#1565c0" if str(pct).startswith('-') else "#333"
    rows += (
        "<tr>"
        + "<td style='padding:8px 12px;border-bottom:1px solid #eee;'>" + label + "</td>"
        + "<td style='padding:8px 12px;border-bottom:1px solid #eee;text-align:right;font-weight:bold;'>" + val + "</td>"
        + "<td style='padding:8px 12px;border-bottom:1px solid #eee;text-align:right;color:" + color + ";'>" + change + "</td>"
        + "<td style='padding:8px 12px;border-bottom:1px solid #eee;text-align:right;color:" + color + ";font-weight:bold;'>" + pct + "</td>"
        + "</tr>"
    )

# ========================================
# HTML組み立て
# ========================================
def build_daytrade_table(results, earnings_flags, mismatch_codes=None):
    if mismatch_codes is None: mismatch_codes = set()
    thead = (
        "<tr style='background:#f5f5f5;'>"
        "<th style='padding:6px 4px;text-align:left;font-size:11px;width:35%;'>銘柄</th>"
        "<th style='padding:6px 4px;text-align:right;font-size:11px;white-space:nowrap;'>株価</th>"
        "<th style='padding:6px 4px;text-align:right;font-size:11px;white-space:nowrap;'>状態</th>"
        "<th style='padding:6px 4px;text-align:right;font-size:11px;white-space:nowrap;'>損切/利確</th>"
        "<th style='padding:6px 4px;text-align:right;font-size:11px;white-space:nowrap;'>ATR</th>"
        "</tr>"
    )
    tbody = ""
    for i, d in enumerate(results):
        earnings   = earnings_flags.get(d['code'], "")
        earnings_html  = ("<br><span style='color:#e65100;font-size:10px;font-weight:bold;'>&#9888;" + earnings + "</span>") if earnings else ""
        mismatch_html  = ("<br><span style='color:#6a1b9a;font-size:10px;font-weight:bold;'>&#9888;目線相違</span>") if d['code'] in mismatch_codes else ""

        price_str = f"{d['price']:,.0f}"
        pct_val   = d['change_pct']
        pct_str   = f"{pct_val:+.2f}%"
        pct_color = "#d32f2f" if pct_val >= 0 else "#1565c0"

        direction = d['dt_direction']
        dir_color = "#d32f2f" if direction == "買い" else "#1565c0"
        dir_label = "↑買い" if direction == "買い" else "↓売り"
        rsi_str   = f"RSI:{d['rsi']:.1f}"

        # 損切・利確表示
        if direction == "買い":
            res_str = f"利: {d['dt_res']:,.0f}" if d['dt_res'] else "利: -"
            sup_str = f"損: {d['dt_sup']:,.0f}" if d['dt_sup'] else "損: -"
        else:
            res_str = f"利: {d['dt_res']:,.0f}" if d['dt_res'] else "利: -"
            sup_str = f"損: {d['dt_sup']:,.0f}" if d['dt_sup'] else "損: -"

        atr_str   = f"{d['atr_val']:,.0f}" if d['atr_val'] > 0 else "-"
        score_str = str(d['daytrade_score'])

        tbody += (
            "<tr>"
            + f"<td style='padding:6px 4px;border-bottom:1px solid #eee;font-size:12px;font-weight:bold;'>"
            + f"{i+1}. {d['name']}<br><span style='font-size:10px;font-weight:normal;color:#666;'>({d['code']})</span>"
            + f"{earnings_html}{mismatch_html}</td>"
            + f"<td style='padding:6px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:12px;font-weight:bold;'>{price_str}</div>"
            + f"<div style='font-size:10px;color:{pct_color};font-weight:bold;'>{pct_str}</div></td>"
            + f"<td style='padding:6px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:12px;font-weight:bold;color:{dir_color};'>{dir_label}</div>"
            + f"<div style='font-size:10px;color:#666;'>{rsi_str}</div></td>"
            + f"<td style='padding:6px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:11px;font-weight:bold;color:#333;'>{res_str}</div>"
            + f"<div style='font-size:10px;color:#666;'>{sup_str}</div></td>"
            + f"<td style='padding:6px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:12px;font-weight:bold;color:#1b5e20;'>{score_str}</div>"
            + f"<div style='font-size:10px;color:#666;'>ATR:{atr_str}</div></td>"
            + "</tr>"
        )
    return (
        "<div style='margin-top:16px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>"
        "<div style='background:#1b5e20;color:#fff;padding:12px 16px;'>"
        "<h2 style='margin:0;font-size:15px;'>今日のデイトレ推奨</h2>"
        "<p style='margin:4px 0 0;font-size:11px;opacity:0.8;'>損切＝-1ATR　利確＝+2ATR</p>"
        "</div><div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;'>"
        "<thead>" + thead + "</thead><tbody>" + tbody + "</tbody></table></div></div>"
    )

def build_swing_table(results, earnings_flags, mismatch_codes=None, market_trend=None):
    if mismatch_codes is None: mismatch_codes = set()
    thead = (
        "<tr style='background:#f5f5f5;'>"
        "<th style='padding:5px 4px;text-align:left;font-size:11px;'>銘柄</th>"
        "<th style='padding:5px 4px;text-align:right;font-size:11px;white-space:nowrap;'>株価</th>"
        "<th style='padding:5px 4px;text-align:right;font-size:11px;white-space:nowrap;'>売買</th>"
        "<th style='padding:5px 4px;text-align:right;font-size:11px;white-space:nowrap;'>目安</th>"
        "<th style='padding:5px 4px;text-align:right;font-size:11px;white-space:nowrap;'>信用</th>"
        "<th style='padding:5px 4px;text-align:right;font-size:11px;white-space:nowrap;'>点数</th>"
        "</tr>"
    )
    tbody = ""
    for i, d in enumerate(results):
        earnings  = earnings_flags.get(d['code'], "")
        flags_html = ""
        if earnings:
            flags_html += f" <span style='color:#e65100;font-size:10px;font-weight:bold;'>&#9888;{earnings}</span>"
        if d['code'] in mismatch_codes:
            flags_html += f" <span style='color:#6a1b9a;font-size:10px;font-weight:bold;'>&#9888;目線相違</span>"

        price_str = f"{d['price']:,.0f}"
        pct_val   = d['change_pct']
        pct_str   = f"{pct_val:+.2f}%"
        pct_color = "#d32f2f" if pct_val >= 0 else "#1565c0"

        direction = d['swing_direction']
        if direction == "買い":
            dir_color = "#d32f2f"; dir_label = "↑買い"; score_color = "#b71c1c"
        else:
            dir_color = "#1565c0"; dir_label = "↓売り"; score_color = "#0d47a1"

        # 状態欄：方向 + RSIのみ（トレンド向きは削除）
        rsi_str = f"RSI:{d['rsi']:.1f}"

        res_str = f"利:{d['swing_res']:,.0f}" if d['swing_res'] else "利:-"
        sup_str = f"損:{d['swing_sup']:,.0f}" if d['swing_sup'] else "損:-"

        score_str = str(d['swing_score'])
        rr_str    = f"RR:{d['swing_rr']:.1f}" if d['swing_rr'] > 0 else "RR:-"
        macd_mark = "&#9733;" if d.get('macd_score', 0) > 0 else ""

        # 信用倍率表示
        ratio = d.get('shinyo_ratio')
        if ratio is None:
            shinyo_str   = "-"
            shinyo_color = "#999"
            shinyo_note  = ""
        else:
            shinyo_str = f"{ratio:.1f}倍"
            if direction == "買い" and ratio >= 5.0:
                shinyo_color = "#d32f2f"
                shinyo_note  = "<div style='font-size:9px;color:#d32f2f;'>▼-10pt</div>"
            elif direction == "売り" and ratio <= 0.5:
                shinyo_color = "#1565c0"
                shinyo_note  = "<div style='font-size:9px;color:#1565c0;'>▼-10pt</div>"
            else:
                shinyo_color = "#333"
                shinyo_note  = ""

        tbody += (
            "<tr>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;font-size:12px;font-weight:bold;white-space:nowrap;'>"
            + f"{i+1}. {d['name']}"
            + f"<span style='font-size:10px;font-weight:normal;color:#888;'> {d['code']}</span>"
            + flags_html + "</td>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:12px;font-weight:bold;'>{price_str}</div>"
            + f"<div style='font-size:10px;color:{pct_color};font-weight:bold;'>{pct_str}</div></td>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:12px;font-weight:bold;color:{dir_color};'>{dir_label}</div>"
            + f"<div style='font-size:10px;color:#666;'>{rsi_str}</div></td>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:11px;font-weight:bold;color:#333;'>{res_str}</div>"
            + f"<div style='font-size:10px;color:#666;'>{sup_str}</div></td>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:12px;font-weight:bold;color:{shinyo_color};'>{shinyo_str}</div>"
            + shinyo_note + "</td>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:12px;font-weight:bold;color:{score_color};'>{score_str}</div>"
            + f"<div style='font-size:10px;color:#666;'>{rr_str}"
            + (f" <span style='color:#ff6f00;'>{macd_mark}MACD</span>" if macd_mark else "")
            + f"</div></td>"
            + "</tr>"
        )
    return (
        "<div style='margin-top:16px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>"
        "<div style='background:#1a237e;color:#fff;padding:12px 16px;'>"
        "<h2 style='margin:0;font-size:15px;'>今週のスイング推奨</h2>"
        "<p style='margin:4px 0 0;font-size:11px;opacity:0.8;'>トレンド20(絶対)・RSI35・%B20・RR15・MACD10</p>"
        + (f"<p style='margin:4px 0 0;font-size:11px;opacity:0.9;'>📊 市場環境フィルター: 日経{'↑上昇' if market_trend=='買い' else '↓下降'}トレンド（{'買い' if market_trend=='買い' else '売り'}シグナルのみ）</p>"
           if market_trend else
           "<p style='margin:4px 0 0;font-size:11px;opacity:0.7;'>📊 市場環境フィルター: データ取得失敗（フィルターなし）</p>")
        + "</div><div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;'>"
        "<thead>" + thead + "</thead><tbody>" + tbody + "</tbody></table></div></div>"
    )

dt_section    = build_daytrade_table(dt_top10,    earnings_flags, MISMATCH_CODES)
swing_section = build_swing_table(swing_top10, earnings_flags, MISMATCH_CODES, market_trend=nikkei_market_trend)

html = (
    "<html><body style='font-family:sans-serif;background:#f5f5f5;padding:12px;'>"
    "<div style='max-width:600px;margin:0 auto;'>"
    "<div style='background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>"
    "<div style='background:#1a237e;color:#fff;padding:16px 20px;'>"
    "<h2 style='margin:0;font-size:18px;'>最新マーケットデータ</h2>"
    "<p style='margin:4px 0 0;font-size:13px;opacity:0.8;'>" + now_str + " JST</p>"
    "</div>"
    "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
    "<thead><tr style='background:#e8eaf6;'>"
    "<th style='padding:8px 12px;text-align:left;'>指標</th>"
    "<th style='padding:8px 12px;text-align:right;'>現在値</th>"
    "<th style='padding:8px 12px;text-align:right;'>前日比</th>"
    "<th style='padding:8px 12px;text-align:right;'>騰落率</th>"
    "</tr></thead>"
    "<tbody>" + rows + "</tbody></table>"
    "<p style='padding:12px 16px;font-size:11px;color:#999;margin:0;'>投資判断はご自身の責任でお願いします</p>"
    "</div>"
    + dt_section
    + swing_section
    + "</div></body></html>"
)

msg = MIMEMultipart('alternative')
msg['From']    = GMAIL_ADDRESS
msg['To']      = SEND_TO
msg['Subject'] = "Souba Data " + now_str
msg.attach(MIMEText(html, 'html', 'utf-8'))

try:
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
        smtp.sendmail(GMAIL_ADDRESS, SEND_TO, msg.as_bytes())
    print("メール送信完了！")
except Exception as e:
    print(f"メール送信失敗: {e}")
