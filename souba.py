import yfinance as yf
import pandas as pd
import duckdb
from pathlib import Path
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
import json

DB_PATH = Path(__file__).parent.parent / "stock_db" / "stock_prices.duckdb"
_HERE = Path(__file__).parent

# ── pandas_ta 代替実装（Python 3.14対応） ──────────────────────
def _rsi_series(closes, period=14):
    arr = np.asarray(closes, float)
    out = np.full(len(arr), np.nan)
    if len(arr) < period + 2:
        return pd.Series(out, index=closes.index)
    d = np.diff(arr)
    g, l = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = g[:period].mean(), l[:period].mean()
    out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
        out[i + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return pd.Series(out, index=closes.index)

def _atr_series(highs, lows, closes, period=14):
    h, l, c = np.asarray(highs, float), np.asarray(lows, float), np.asarray(closes, float)
    out = np.full(len(h), np.nan)
    if len(h) < period + 1:
        return pd.Series(out, index=highs.index)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    v = tr[:period].mean()
    out[period] = v
    for i in range(period, len(tr)):
        v = (v * (period - 1) + tr[i]) / period
        out[i + 1] = v
    return pd.Series(out, index=highs.index)

def _bbands(closes, period=20, std=2):
    arr = closes.values.astype(float)
    lo, hi = np.full(len(arr), np.nan), np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        sl = arr[i - period + 1:i + 1]
        m, s = sl.mean(), sl.std(ddof=0)
        lo[i], hi[i] = m - std * s, m + std * s
    return pd.Series(lo, index=closes.index), pd.Series(hi, index=closes.index)

def _ema(arr, period):
    arr = np.asarray(arr, float)
    if len(arr) < period:
        return np.array([])
    k, out = 2.0 / (period + 1), [arr[:period].mean()]
    for v in arr[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return np.array(out)

def _macd_series(closes, fast=12, slow=26, signal=9):
    arr = closes.values.astype(float)
    idx = closes.index
    empty = pd.Series(np.full(len(arr), np.nan), index=idx)
    if len(arr) < slow + signal:
        return empty, empty
    e12, e26 = _ema(arr, fast), _ema(arr, slow)
    macd = e26 - e12[fast - slow:]
    if len(macd) < signal:
        return empty, empty
    sig = _ema(macd, signal)
    macd_out = np.full(len(arr), np.nan)
    sig_out  = np.full(len(arr), np.nan)
    macd_out[slow - 1:] = macd
    sig_out[slow + signal - 2:] = sig
    return pd.Series(macd_out, index=idx), pd.Series(sig_out, index=idx)
# ─────────────────────────────────────────────────────────────

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
# 【変更①】信用倍率：JSONキャッシュから読み込み
# ========================================
def load_shinyo_cache():
    """
    shinyo_cache.jsonを読み込む。
    ファイルが存在しない場合は空dictを返す。
    """
    try:
        with open(_HERE / "shinyo_cache.json", "r", encoding="utf-8") as f:
            cache = json.load(f)
        updated_at = cache.get("updated_at", "不明")
        data = cache.get("data", {})
        print(f"信用倍率キャッシュ読み込み: {len(data)}銘柄 (更新日時: {updated_at})")
        return data
    except FileNotFoundError:
        print("shinyo_cache.json が見つかりません。信用倍率なしで実行します。")
        return {}
    except Exception as e:
        print(f"信用倍率キャッシュ読み込みエラー: {e}")
        return {}

shinyo_data = load_shinyo_cache()

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
# 【変更②】STEP1：最低売買代金フィルター（上位50廃止）
# 売買代金10億円/日以上の銘柄を全て対象にする
# ========================================
MIN_TRADING_VALUE = 10  # 億円/日

print(f"STEP 1：売買代金{MIN_TRADING_VALUE}億円以上の銘柄を抽出中...")

_con = duckdb.connect(str(DB_PATH), read_only=True)
_codes_str = ",".join(f"'{c}'" for c in nikkei225_codes)
_rows = _con.execute(f"""
    SELECT p.code, (p.close * p.volume / 1e8) AS trading_value
    FROM prices p
    INNER JOIN (
        SELECT code, MAX(date) AS max_date
        FROM prices
        WHERE code IN ({_codes_str})
        GROUP BY code
    ) latest ON p.code = latest.code AND p.date = latest.max_date
    WHERE (p.close * p.volume / 1e8) >= {MIN_TRADING_VALUE}
""").fetchall()
_con.close()
filtered_stocks = [{'code': r[0], 'trading_value': r[1]} for r in _rows]

# 名前取得
candidate_stocks = []
for v in filtered_stocks:
    name = get_japanese_name(v['code'])
    candidate_stocks.append((v['code'], name))
    time.sleep(0.1)

print(f"対象銘柄: {len(candidate_stocks)}銘柄（売買代金{MIN_TRADING_VALUE}億円以上）")

# ========================================
# 【変更③】サポレジ改善：2年データ×反転回数複合HVN
# ========================================
def calc_support_resistance(df, curr, direction, atr_val):
    """
    過去2年の日足データから価格帯別出来高×反転回数でサポレジを検出。

    direction: 'buy' or 'sell'
    戻り値:
        sl_price: 損切りライン
        tp_price: 利確ライン
        sl_strength: ラインの強度（★数）
        tp_strength: ラインの強度（★数）
    """
    try:
        if len(df) < 60:
            return None, None, 0, 0

        close = df['Close'].values
        high  = df['High'].values
        low   = df['Low'].values
        vol   = df['Volume'].values
        n     = len(df)

        # ---- ①価格帯別出来高（bins=20・直近3ヶ月×2重み） ----
        price_min = close.min() * 0.98
        price_max = close.max() * 1.02
        bins = 20
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 直近3ヶ月（約63日）は重み2倍
        cutoff = max(0, n - 63)
        weights = np.ones(n)
        weights[cutoff:] = 2.0

        vol_by_bin = np.zeros(bins)
        for i in range(n):
            idx = np.searchsorted(bin_edges[1:], close[i])
            idx = min(idx, bins - 1)
            vol_by_bin[idx] += vol[i] * weights[i]

        # ---- ②反転回数カウント ----
        # 各価格帯で高値・安値が何回その帯に入ったかをカウント
        reversal_by_bin = np.zeros(bins)
        for i in range(1, n - 1):
            # 高値反転（直前・直後より高い）
            if high[i] > high[i-1] and high[i] > high[i+1]:
                idx = np.searchsorted(bin_edges[1:], high[i])
                idx = min(idx, bins - 1)
                reversal_by_bin[idx] += 1
            # 安値反転（直前・直後より低い）
            if low[i] < low[i-1] and low[i] < low[i+1]:
                idx = np.searchsorted(bin_edges[1:], low[i])
                idx = min(idx, bins - 1)
                reversal_by_bin[idx] += 1

        # ---- ③複合スコア（出来高×反転回数） ----
        vol_norm      = vol_by_bin / (vol_by_bin.max() + 1e-10)
        reversal_norm = reversal_by_bin / (reversal_by_bin.max() + 1e-10)
        composite     = vol_norm * 0.6 + reversal_norm * 0.4

        # ---- ④現在値の上下に分けてTop候補を選ぶ ----
        curr_bin = np.searchsorted(bin_edges[1:], curr)
        curr_bin = min(curr_bin, bins - 1)

        # ATRベースの最小距離制約
        # SL: 0.5ATR以上離れたサポートのみ（近すぎるストップを防ぐ）
        # TP: 1.5ATR以上離れたレジスタンスのみ（RR≥1.5を構造的に担保）
        min_sl_dist = atr_val * 0.5
        min_tp_dist = atr_val * 1.5

        # 下側：サポート候補
        support_bins = [(i, composite[i], bin_centers[i])
                        for i in range(curr_bin)
                        if bin_centers[i] < curr - min_sl_dist]
        support_bins.sort(key=lambda x: x[1], reverse=True)

        # 上側：レジスタンス候補
        resist_bins = [(i, composite[i], bin_centers[i])
                       for i in range(curr_bin + 1, bins)
                       if bin_centers[i] > curr + min_tp_dist]
        resist_bins.sort(key=lambda x: x[1], reverse=True)

        # ---- ⑤方向で損切り・利確を割り当て ----
        def strength(score):
            if score >= 0.7: return 3
            if score >= 0.4: return 2
            return 1

        if direction == 'buy':
            # 買い：直下サポート=損切り、直上レジスタンス=利確
            sl_price    = support_bins[0][2]  if support_bins else curr - atr_val * 1.5
            sl_strength = strength(support_bins[0][1]) if support_bins else 1
            tp_price    = resist_bins[0][2]   if resist_bins  else curr + atr_val * 3.0
            tp_strength = strength(resist_bins[0][1]) if resist_bins else 1
        else:
            # 売り：直上レジスタンス=損切り、直下サポート=利確
            sl_price    = resist_bins[0][2]   if resist_bins  else curr + atr_val * 1.5
            sl_strength = strength(resist_bins[0][1]) if resist_bins else 1
            tp_price    = support_bins[0][2]  if support_bins else curr - atr_val * 3.0
            tp_strength = strength(support_bins[0][1]) if support_bins else 1

        return sl_price, tp_price, sl_strength, tp_strength

    except Exception as e:
        print(f"サポレジ計算エラー: {e}")
        return None, None, 0, 0


# ========================================
# MACDクロス鮮度チェック
# ========================================
def calc_macd_cross(df, direction):
    try:
        macd_line, signal_line = _macd_series(df['Close'], fast=12, slow=26, signal=9)
        macd_line, signal_line = macd_line.dropna(), signal_line.dropna()
        if len(macd_line) < 4: return 0
        for i in range(-3, 0):
            prev_diff = macd_line.iloc[i-1] - signal_line.iloc[i-1]
            curr_diff = macd_line.iloc[i]   - signal_line.iloc[i]
            golden = prev_diff < 0 and curr_diff >= 0
            dead   = prev_diff > 0 and curr_diff <= 0
            if direction == "買い" and golden: return 10
            if direction == "売り" and dead:   return 10
        return 0
    except:
        return 0

# ========================================
# STEP 2：データ取得・raw値収集
# 【変更点】
#  - データ期間を2年に延長
#  - サポレジをcalc_support_resistanceに差し替え
#  - %ATRフィルター（0.8%未満は除外フラグ）
# ========================================
def calc_raw(code, name):
    try:
        # ローカルDBから2年分取得
        _c = duckdb.connect(str(DB_PATH), read_only=True)
        df = _c.execute("""
            SELECT date AS Date, open AS Open, high AS High,
                   low AS Low, close AS Close, volume AS Volume
            FROM prices
            WHERE code = ?
              AND date >= (CURRENT_DATE - INTERVAL '2' YEAR)
            ORDER BY date
        """, [code]).df()
        _c.close()
        if df.empty or len(df) < 60: return None
        df['Date'] = pd.to_datetime(df['Date'])

        df['RSI']  = _rsi_series(df['Close'], period=14)
        df['MA5']  = df['Close'].rolling(5).mean()
        df['MA25'] = df['Close'].rolling(25).mean()
        df['MA75'] = df['Close'].rolling(75).mean()
        atr_series = _atr_series(df['High'], df['Low'], df['Close'], period=14)

        recent = df.tail(20)
        latest = df.iloc[-1]
        curr   = latest['Close']
        rsi    = latest['RSI'] if not pd.isna(latest['RSI']) else 50

        change_pct = ((curr - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100 if len(df) >= 2 else 0.0)

        ma5  = latest['MA5']
        ma25 = latest['MA25']
        ma75 = latest['MA75']
        ma_divergence = (ma25 - ma75) / ma75 * 100 if not pd.isna(ma25) and not pd.isna(ma75) and ma75 != 0 else 0.0
        trend = '上昇' if ma_divergence > 0 else '下降'
        ma5_25_bull = (bool(ma5 >= ma25) if not pd.isna(ma5) and not pd.isna(ma25) else False)
        atr_val = atr_series.dropna().iloc[-1] if atr_series is not None and not atr_series.isna().all() else 0

        # 【変更④】%ATRフィルター：0.8%未満はスイング除外フラグ
        atr_pct = (atr_val / curr * 100) if curr > 0 else 0
        low_volatility = atr_pct < 0.8

        # ---- デイトレ：ATRベース損切・利確 ----
        dt_direction = "買い" if change_pct >= 0 else "売り"
        if dt_direction == "買い":
            dt_sup = curr - atr_val
            dt_res = curr + 2 * atr_val
        else:
            dt_sup = curr + atr_val
            dt_res = curr - 2 * atr_val
        dt_rr_valid = atr_val > 0

        # ---- 【変更③】スイング：サポレジ差し替え ----
        buy_sl,  buy_tp,  buy_sl_str,  buy_tp_str  = calc_support_resistance(df, curr, 'buy',  atr_val)
        sell_sl, sell_tp, sell_sl_str, sell_tp_str = calc_support_resistance(df, curr, 'sell', atr_val)

        # RR計算
        rr_buy = rr_sell = 0.0
        if buy_tp is not None and buy_sl is not None and curr != buy_sl:
            rr_buy  = abs(buy_tp  - curr) / abs(curr - buy_sl)
        if sell_tp is not None and sell_sl is not None and curr != sell_sl:
            rr_sell = abs(curr - sell_tp) / abs(sell_sl - curr)

        # ---- ボリバン %B ----
        pct_b = None
        try:
            bb_lower, bb_upper = _bbands(df['Close'], period=20, std=2)
            lower, upper = bb_lower.iloc[-1], bb_upper.iloc[-1]
            if not np.isnan(lower) and not np.isnan(upper) and upper != lower:
                pct_b = (curr - lower) / (upper - lower)
        except:
            pass

        # ---- デイトレスコア ----
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

        # RSIスコア
        if 40 <= rsi <= 60:
            buy_rsi_score = 35; sell_rsi_score = 35
        elif 30 <= rsi < 40:
            buy_rsi_score = 28; sell_rsi_score = 14
        elif 60 < rsi <= 70:
            buy_rsi_score = 14; sell_rsi_score = 28
        elif rsi < 30:
            buy_rsi_score = 35; sell_rsi_score = 0
        else:
            buy_rsi_score = 0;  sell_rsi_score = 35

        return {
            'code':             code,
            'name':             name,
            'price':            curr,
            'change_pct':       change_pct,
            'rsi':              rsi,
            'trend':            trend,
            'ma_divergence':    ma_divergence,
            'ma5_25_bull':      ma5_25_bull,
            'atr_pct':          atr_pct,
            'low_volatility':   low_volatility,
            # デイトレ
            'dt_direction':   dt_direction,
            'dt_sup':         dt_sup,
            'dt_res':         dt_res,
            'dt_rr_valid':    dt_rr_valid,
            'atr_val':        atr_val,
            'daytrade_score': round(daytrade_score, 1),
            # スイング用raw
            'rr_buy_raw':     rr_buy,
            'rr_sell_raw':    rr_sell,
            'buy_rsi_score':  buy_rsi_score,
            'sell_rsi_score': sell_rsi_score,
            'buy_sl_price':   buy_sl,
            'buy_tp_price':   buy_tp,
            'sell_sl_price':  sell_sl,
            'sell_tp_price':  sell_tp,
            'buy_sl_str':     buy_sl_str,
            'buy_tp_str':     buy_tp_str,
            'sell_sl_str':    sell_sl_str,
            'sell_tp_str':    sell_tp_str,
            'pct_b':          pct_b,
            '_df':            df,
        }
    except Exception as e:
        print(f"エラー {code}: {e}")
        return None

print("STEP 2：全銘柄rawデータ収集中...")
all_results = []
for code, name in candidate_stocks:
    r = calc_raw(code, name)
    if r: all_results.append(r)
    time.sleep(0.1)

# ========================================
# スイングスコア確定
# ========================================
def rr_score_abs(rr):
    """
    RR1.5〜2.5を最高点、外れるほど下げる設計。
    RR<1.0は0点（TP<SL距離のトレードは除外）。
    RR3.5+は3点に降格（TP遠すぎて5日で届かない）。
    """
    if rr is None or rr <= 0: return 0
    if rr < 1.0:  return 0   # RR<1.0は選外
    if rr < 1.5:  return 6
    if rr < 2.5:  return 15  # 最適帯
    if rr < 3.5:  return 9   # やや遠すぎ
    return 3                  # 3.5+は過剰

for i, r in enumerate(all_results):
    # 【スコア改善】乖離率が2%未満は「方向性なし」としてトレンドスコア0pt
    # 弱いトレンドは方向が不確かなため、強い乖離のみ20ptを付与
    div_abs = abs(r['ma_divergence'])
    if div_abs < 2.0:
        buy_trend_sc_i  = 0
        sell_trend_sc_i = 0
    elif r['ma_divergence'] > 0:
        buy_trend_sc_i  = 20
        sell_trend_sc_i = 0
    else:
        buy_trend_sc_i  = 0
        sell_trend_sc_i = 20

    pct_b = r['pct_b']
    if pct_b is not None:
        buy_bb_score  = max(0, round((1 - pct_b) * 20, 1))
        sell_bb_score = max(0, round(pct_b * 20, 1))
    else:
        buy_bb_score = sell_bb_score = 0

    buy_score_tmp  = buy_trend_sc_i  + r['buy_rsi_score']  + rr_score_abs(r['rr_buy_raw'])  + buy_bb_score
    sell_score_tmp = sell_trend_sc_i + r['sell_rsi_score'] + rr_score_abs(r['rr_sell_raw']) + sell_bb_score
    tentative_dir  = "買い" if buy_score_tmp >= sell_score_tmp else "売り"

    df_tmp     = r.pop('_df')
    macd_score = calc_macd_cross(df_tmp, tentative_dir)

    buy_score  = buy_score_tmp  + (macd_score if tentative_dir == "買い" else 0)
    sell_score = sell_score_tmp + (macd_score if tentative_dir == "売り" else 0)

    # 【スコア改善】逆トレンド方向ペナルティ：上昇中の売り・下降中の買いは-20pt
    # （従来の-10ptより強く抑制し、逆張りシグナルを上位から排除）
    if   tentative_dir == "売り" and r['ma_divergence'] > 0:
        sell_score = max(0, round(sell_score - 20, 1))
    elif tentative_dir == "買い" and r['ma_divergence'] < 0:
        buy_score  = max(0, round(buy_score  - 20, 1))

    buy_rr_valid  = (r['buy_tp_price']  is not None and r['rr_buy_raw']  > 0)
    sell_rr_valid = (r['sell_tp_price'] is not None and r['rr_sell_raw'] > 0)

    RR_TARGET = 1.0  # 目標RR（SL逆算用）

    price = r['price']
    if buy_score >= sell_score:
        tp = r['buy_tp_price']
        tp_dist = (tp - price) if tp is not None else 0
        rr_sl = (price - tp_dist / RR_TARGET) if tp_dist > 0 else None
        r['swing_direction'] = "買い"
        r['swing_score']     = round(buy_score, 1)
        r['swing_sup']       = rr_sl              # RR=1.0逆算SL（実用損切り）
        r['swing_res']       = tp
        r['swing_sup_str']   = 0                  # 計算値なので★なし
        r['swing_res_str']   = r['buy_tp_str']
        r['swing_hvn_sup']   = r['buy_sl_price']  # HVN下値めど（参考）
        r['swing_hvn_str']   = r['buy_sl_str']
        r['swing_rr']        = RR_TARGET
        r['swing_rr_valid']  = buy_rr_valid
    else:
        tp = r['sell_tp_price']
        tp_dist = (price - tp) if tp is not None else 0
        rr_sl = (price + tp_dist / RR_TARGET) if tp_dist > 0 else None
        r['swing_direction'] = "売り"
        r['swing_score']     = round(sell_score, 1)
        r['swing_sup']       = rr_sl              # RR=1.0逆算SL（実用損切り）
        r['swing_res']       = tp
        r['swing_sup_str']   = 0                  # 計算値なので★なし
        r['swing_res_str']   = r['sell_tp_str']
        r['swing_hvn_sup']   = r['sell_sl_price'] # HVN上値めど（参考）
        r['swing_hvn_str']   = r['sell_sl_str']
        r['swing_rr']        = RR_TARGET
        r['swing_rr_valid']  = sell_rr_valid

    r['macd_score']     = macd_score
    r['buy_bb_score']   = buy_bb_score
    r['sell_bb_score']  = sell_bb_score

print("STEP 3：決算日フラグ取得中...")
earnings_flags = {}
for code, name in candidate_stocks:
    flag = get_earnings_flag(code)
    if flag:
        earnings_flags[code] = flag
    time.sleep(0.1)

# 【変更①】信用倍率はJSONから参照済み（STEP4のスクレイピングループを削除）
# shinyo_dataはload_shinyo_cache()で取得済み

# 信用倍率スイングスコア補正
for r in all_results:
    ratio = shinyo_data.get(r['code'])
    r['shinyo_ratio'] = ratio
    if ratio is not None:
        if r['swing_direction'] == "買い" and ratio >= 5.0:
            r['swing_score'] = round(r['swing_score'] - 10, 1)
        elif r['swing_direction'] == "売り" and ratio <= 0.5:
            r['swing_score'] = round(r['swing_score'] - 10, 1)

def get_nikkei_trend(fast=10, slow=25):
    try:
        df = yf.Ticker("^N225").history(period="3mo", auto_adjust=False)
        if df.empty or len(df) < slow:
            return None
        closes = df['Close'].dropna().values.astype(float)  # NaN除去
        if len(closes) < slow:
            return None
        return '買い' if closes[-fast:].mean() >= closes[-slow:].mean() else '売り'
    except:
        return None

def get_nikkei_dual_ma():
    """
    デュアルMAフィルター：MA5/25 AND MA25/75 の両方が一致する方向を返す
    一致：'買い' or '売り'
    不一致（転換期）：'転換期'
    取得失敗：None
    """
    try:
        df = yf.Ticker("^N225").history(period="6mo", auto_adjust=False)
        if df.empty or len(df) < 75:
            return None
        closes = df['Close'].dropna().values.astype(float)  # NaN除去してから計算
        if len(closes) < 75:
            return None
        ma5  = closes[-5:].mean()    # 直近5日移動平均
        ma25 = closes[-25:].mean()   # 直近25日移動平均
        ma75 = closes[-75:].mean()   # 直近75日移動平均
        trend_short = '買い' if ma5  >= ma25 else '売り'
        trend_long  = '買い' if ma25 >= ma75 else '売り'
        if trend_short == trend_long:
            return trend_short   # 両方一致
        else:
            return '転換期'      # 不一致→休む
    except:
        return None

# ========================================
# TOP10選出
# 【変更④】%ATRフィルター：低ボラ銘柄をスイング対象から除外
# 【変更⑤】スコア最低閾値：60点未満は除外（弱い売りシグナルを自然淘汰）
# 【改善】MIN_DIVERGENCE：MA乖離率2%未満は「方向性不明」として除外
# ========================================
MIN_SWING_SCORE = 60
MIN_DIVERGENCE  = 2.0   # MA25/75乖離率の最低ライン（絶対値）

dt_top10 = sorted(all_results, key=lambda x: x['daytrade_score'], reverse=True)[:10]

swing_valid = [
    r for r in all_results
    if r['swing_rr_valid']
    and abs(r['change_pct']) <= 3.0
    and not r['low_volatility']        # 【変更④】%ATR 0.8%未満を除外
    and r['swing_score'] >= MIN_SWING_SCORE  # 【変更⑤】スコア60点未満を除外
    and abs(r['ma_divergence']) >= MIN_DIVERGENCE  # 【改善】乖離率2%未満を除外
]

nikkei_market_trend = get_nikkei_trend(fast=10, slow=25)
nikkei_dual_ma      = get_nikkei_dual_ma()

# ハイブリッドMAフィルター適用
# 転換期（不一致）→ 銘柄デュアルMAフォールバック（各銘柄のMA5/25 AND MA25/75 が一致する銘柄のみ、方向フリー）
# 明確トレンド（一致）→ 順張りは無条件通過、逆張りも個別デュアルMAが揃えば通過（-10ptペナルティあり）
# 取得失敗 → フィルターなし（従来通り）
def _stock_dual_ma_ok(r, direction):
    """銘柄レベルのデュアルMAが指定方向と一致するか"""
    ma25_75_ok = (r['ma_divergence'] > 0) if direction == '買い' else (r['ma_divergence'] < 0)
    ma5_25_ok  = r.get('ma5_25_bull', False) if direction == '買い' else (not r.get('ma5_25_bull', True))
    return ma25_75_ok and ma5_25_ok

if nikkei_dual_ma in ('転換期', '買い', '売り'):
    swing_valid_filtered = []
    for r in swing_valid:
        direction = r['swing_direction']
        if nikkei_dual_ma == '転換期':
            # 転換期：個別MA5/25のみで判定（MA25/75は不要）
            # 理由：回復途上の銘柄はMA5>MA25（短期回復）だがMA25<MA75（長期まだ戻ってない）
            # →MA25/75を要求すると買いシグナルが全滅し売りだらけになるため
            ma5_25_ok = r.get('ma5_25_bull', False) if direction == '買い' \
                        else (not r.get('ma5_25_bull', True))
            if ma5_25_ok:
                swing_valid_filtered.append(r)
        elif direction == nikkei_dual_ma:
            # 順張り：無条件通過
            swing_valid_filtered.append(r)
        else:
            # 逆張り：個別デュアルMAが揃っている場合のみ通過（-10ptペナルティは下の処理で付与）
            if _stock_dual_ma_ok(r, direction):
                swing_valid_filtered.append(r)
else:
    swing_valid_filtered = swing_valid  # 取得失敗時はフィルターなし

# 既存の逆張りペナルティ（デュアルMAが有効な場合のみ適用）
if nikkei_market_trend is not None:
    for r in swing_valid_filtered:
        if r['swing_direction'] != nikkei_market_trend:
            r['swing_score'] = round(r['swing_score'] - 10, 1)
            r['market_counter_trend'] = True
        else:
            r['market_counter_trend'] = False
else:
    for r in swing_valid_filtered:
        r['market_counter_trend'] = False

# ── 変調検知：通過件数が10件未満なら敗者復活（逆方向×MA5/25一致）で補充 ──
HENSHO_THRESHOLD = 10
swing_hensho_triggered = False

if (len(swing_valid_filtered) < HENSHO_THRESHOLD
        and nikkei_dual_ma in ('買い', '売り')):          # 転換期は除外（既に個別MAフォールバック）
    swing_hensho_triggered = True
    counter_dir = '買い' if nikkei_dual_ma == '売り' else '売り'
    needed      = HENSHO_THRESHOLD - len(swing_valid_filtered)
    already     = {r['code'] for r in swing_valid_filtered}

    repechage_candidates = []
    # 敗者復活専用pool：逆方向銘柄のみスコア・乖離率を緩和（通常フィルターは影響しない）
    repechage_pool = [
        r for r in all_results
        if r['code'] not in already
        and r['swing_direction'] == counter_dir   # 逆方向のみ
        and r.get('swing_rr_valid')
        and abs(r.get('change_pct', 99)) <= 3.0
        and not r.get('low_volatility', True)
        and r.get('swing_score', 0) >= 50         # 通常60→敗者復活は50
        and abs(r.get('ma_divergence', 0)) >= 1.0 # 通常2%→敗者復活は1%
    ]
    for r in repechage_pool:
        # MA5/25のみチェック（MA25/75は不問）
        ma5_25_ok = r.get('ma5_25_bull', False) if counter_dir == '買い' \
                    else (not r.get('ma5_25_bull', True))
        if ma5_25_ok:
            r['market_counter_trend'] = True
            r['swing_score']          = round(r['swing_score'] - 10, 1)
            r['repechage']            = True
            repechage_candidates.append(r)

    repechage_candidates.sort(key=lambda x: x['swing_score'], reverse=True)
    swing_valid_filtered = swing_valid_filtered + repechage_candidates[:needed]

swing_top10 = sorted(swing_valid_filtered, key=lambda x: x['swing_score'], reverse=True)[:10]

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
def strength_stars(n):
    return '★' * n + '☆' * (3 - n)

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
        earnings      = earnings_flags.get(d['code'], "")
        earnings_html = ("<br><span style='color:#e65100;font-size:10px;font-weight:bold;'>&#9888;" + earnings + "</span>") if earnings else ""
        mismatch_html = ("<br><span style='color:#6a1b9a;font-size:10px;font-weight:bold;'>&#9888;目線相違</span>") if d['code'] in mismatch_codes else ""

        price_str = f"{d['price']:,.0f}"
        pct_val   = d['change_pct']
        pct_str   = f"{pct_val:+.2f}%"
        pct_color = "#d32f2f" if pct_val >= 0 else "#1565c0"

        direction = d['dt_direction']
        dir_color = "#d32f2f" if direction == "買い" else "#1565c0"
        dir_label = "↑買い" if direction == "買い" else "↓売り"
        rsi_str   = f"RSI:{d['rsi']:.1f}"

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

def build_swing_table(results, earnings_flags, mismatch_codes=None, market_trend=None, dual_ma_status=None,
                      hensho_triggered=False, all_filtered=None):
    if mismatch_codes is None: mismatch_codes = set()

    # シグナルなしの場合は専用メッセージを返す（転換期でも銘柄MAで候補があれば通常表示）
    if not results:
        return (
            "<div style='margin-top:16px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>"
            "<div style='background:#1a237e;color:#fff;padding:12px 16px;'>"
            "<h2 style='margin:0;font-size:15px;'>今週のスイング推奨</h2>"
            "</div>"
            "<div style='padding:20px 16px;text-align:center;'>"
            "<p style='font-size:12px;color:#666;margin:0;'>本日は条件を満たす銘柄がありませんでした。</p>"
            "</div></div>"
        )

    thead = (
        "<tr style='background:#f5f5f5;'>"
        "<th style='padding:5px 4px;text-align:left;font-size:11px;width:26%;'>銘柄</th>"
        "<th style='padding:5px 4px;text-align:right;font-size:11px;white-space:nowrap;'>株価</th>"
        "<th style='padding:5px 4px;text-align:right;font-size:11px;white-space:nowrap;'>売買</th>"
        "<th style='padding:5px 4px;text-align:right;font-size:11px;white-space:nowrap;'>目安</th>"
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

        rsi_str = f"RSI:{d['rsi']:.1f}"

        # ラインの強度表示（★）
        res_str = f"利:{d['swing_res']:,.0f} {strength_stars(d.get('swing_res_str',1))}" if d['swing_res'] else "利:-"
        # 損切り（RR=1.0逆算）
        sup_str  = f"損:{d['swing_sup']:,.0f}" if d['swing_sup'] else "損:-"
        # HVN下値めど（参考）
        hvn_val  = d.get('swing_hvn_sup')
        hvn_str  = f"めど:{hvn_val:,.0f} {strength_stars(d.get('swing_hvn_str',1))}" if hvn_val else ""

        score_str    = str(d['swing_score'])
        # TP距離を%で表示（RR=1.0なので損切りも同距離）
        tp_pct_str   = f"TP:{abs(d['swing_res']-d['price'])/d['price']*100:.1f}%" if d['swing_res'] else ""
        atr_pct_str  = f"%ATR:{d['atr_pct']:.1f}%"
        macd_mark     = "&#9733;" if d.get('macd_score', 0) > 0 else ""
        counter_mark  = "&#9660;-10pt逆張り" if d.get('market_counter_trend') else ""
        repechage_mark = d.get('repechage', False)

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

        row_bg = "background:#fffde7;" if repechage_mark else ""
        tbody += (
            f"<tr style='{row_bg}'>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;font-size:12px;font-weight:bold;white-space:nowrap;'>"
            + (f"<span style='color:#f57f17;font-size:10px;'>&#9889;</span>" if repechage_mark else "")
            + f"{i+1}. {d['name']}"
            + f"<span style='font-size:10px;font-weight:normal;color:#888;'> {d['code']}</span>"
            + flags_html + "</td>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:12px;font-weight:bold;'>{price_str}</div>"
            + f"<div style='font-size:10px;color:{pct_color};font-weight:bold;'>{pct_str}</div></td>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:12px;font-weight:bold;color:{dir_color};'>{dir_label}</div>"
            + f"<div style='font-size:10px;color:#666;'>{rsi_str}</div>"
            + f"<div style='font-size:10px;font-weight:bold;color:{shinyo_color};'>{shinyo_str}</div>"
            + shinyo_note + "</td>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:11px;font-weight:bold;color:#333;'>{res_str}</div>"
            + f"<div style='font-size:10px;color:#c62828;'>{sup_str}</div>"
            + (f"<div style='font-size:9px;color:#999;'>{hvn_str}</div>" if hvn_str else "")
            + "</td>"
            + f"<td style='padding:5px 4px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap;'>"
            + f"<div style='font-size:12px;font-weight:bold;color:{score_color};'>{score_str}</div>"
            + f"<div style='font-size:10px;color:#666;'>{tp_pct_str}"
            + (f" <span style='color:#ff6f00;'>{macd_mark}MACD</span>" if macd_mark else "")
            + f"</div>"
            + f"<div style='font-size:9px;color:#888;'>{atr_pct_str}"
            + (f" {counter_mark}" if counter_mark else "")
            + f"</div></td>"
            + "</tr>"
        )
    return (
        "<div style='margin-top:16px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>"
        "<div style='background:#1a237e;color:#fff;padding:12px 16px;'>"
        "<h2 style='margin:0;font-size:15px;'>今週のスイング推奨</h2>"
        "<p style='margin:4px 0 0;font-size:11px;opacity:0.8;'>トレンド20(絶対)・RSI35・%B20・RR15・MACD10｜サポレジ2年HVN</p>"
        + (f"<p style='margin:4px 0 0;font-size:11px;opacity:0.9;'>📊 市場環境: 日経{'↑上昇' if market_trend=='買い' else '↓下降'}トレンド（逆張りは-10pt）</p>"
           if market_trend else
           "<p style='margin:4px 0 0;font-size:11px;opacity:0.7;'>📊 市場環境: データ取得失敗（ペナルティなし）</p>")
        + (f"<p style='margin:4px 0 0;font-size:12px;background:rgba(255,150,0,0.3);border-radius:4px;padding:4px 8px;'>⚠️ 日経転換期: MA5/25≠MA25/75 — 個別銘柄MAが揃った銘柄のみ表示</p>"
           if nikkei_dual_ma == '転換期' else
           f"<p style='margin:4px 0 0;font-size:11px;opacity:0.9;'>✅ 日経デュアルMA: {'↑上昇' if nikkei_dual_ma=='買い' else '↓下降'}トレンド一致（{nikkei_dual_ma}順張り通過・逆張りは個別MA揃いのみ）</p>"
           if nikkei_dual_ma in ('買い','売り') else "")
        + (f"<p style='margin:4px 0 0;font-size:12px;background:rgba(255,193,7,0.35);border-radius:4px;padding:4px 8px;font-weight:bold;'>&#9889; 変調検知: 通常通過{len([r for r in (all_filtered or []) if not r.get('repechage')])}件 &mdash; &#9889;マーク銘柄は逆方向敗者復活</p>"
           if hensho_triggered else "")
        + "</div><div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;'>"
        "<thead>" + thead + "</thead><tbody>" + tbody + "</tbody></table></div></div>"
    )

dt_section    = build_daytrade_table(dt_top10,    earnings_flags, MISMATCH_CODES)
swing_section = build_swing_table(swing_top10, earnings_flags, MISMATCH_CODES,
                                   market_trend=nikkei_market_trend, dual_ma_status=nikkei_dual_ma,
                                   hensho_triggered=swing_hensho_triggered, all_filtered=swing_valid_filtered)

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

# ========================================
# JSON保存（GitHub Pages Webアプリ用）
# ========================================
def save_json_results():
    data_dir = _HERE / "docs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d')

    def _swing_entry(r):
        tp   = r.get('swing_res')
        sl   = r.get('swing_sup')
        price = r['price']
        tp_pct = round((tp - price) / price * 100, 2) if tp and price else None
        sl_pct = round((sl - price) / price * 100, 2) if sl and price else None
        return {
            'code':          r['code'],
            'name':          r['name'],
            'price':         round(float(price), 1) if price else None,
            'change_pct':    round(float(r.get('change_pct', 0)), 2),
            'direction':     r['swing_direction'],
            'score':         float(r['swing_score']),
            'rsi':           round(float(r.get('rsi', 0)), 1),
            'shinyo_ratio':  r.get('shinyo_ratio'),
            'tp_price':      round(float(tp), 1) if tp else None,
            'sl_price':      round(float(sl), 1) if sl else None,
            'tp_pct':        tp_pct,
            'sl_pct':        sl_pct,
            'tp_strength':   int(r.get('swing_res_str', 0) or 0),
            'hvn_price':     round(float(r['swing_hvn_sup']), 1) if r.get('swing_hvn_sup') else None,
            'hvn_strength':  int(r.get('swing_hvn_str', 0) or 0),
            'atr_pct':       round(float(r.get('atr_pct', 0)), 2),
            'ma_divergence': round(float(r.get('ma_divergence', 0)), 2),
            'counter_trend': bool(r.get('market_counter_trend', False)),
            'repechage':     bool(r.get('repechage', False)),
            'macd_signal':   bool(r.get('macd_score', 0) > 0),
        }

    def _dt_entry(r):
        return {
            'code':       r['code'],
            'name':       r['name'],
            'price':      round(float(r['price']), 1),
            'change_pct': round(float(r.get('change_pct', 0)), 2),
            'direction':  r.get('dt_direction', ''),
            'score':      float(r['daytrade_score']),
            'rsi':        round(float(r.get('rsi', 0)), 1),
            'atr_pct':    round(float(r.get('atr_pct', 0)), 2),
        }

    result = {
        'date':        today,
        'generated_at': now_str,
        'nikkei': {
            'close':        round(float(nikkei), 1) if nikkei else None,
            'change_pct':   nk_p,
            'ma_status':    nikkei_dual_ma or 'N/A',
            'market_trend': nikkei_market_trend or 'N/A',
        },
        'market': {
            'dow':    {'value': round(float(dow),    1) if dow    else None, 'change_pct': dow_p},
            'sp500':  {'value': round(float(sp),     1) if sp     else None, 'change_pct': sp_p},
            'nasdaq': {'value': round(float(nasdaq), 1) if nasdaq else None, 'change_pct': nasdaq_p},
            'usdjpy': {'value': round(float(usdjpy), 3) if usdjpy else None, 'change_pct': fx_p},
            'vix':    {'value': round(float(vix),    2) if vix    else None, 'change_pct': vix_p},
        },
        'hensho_triggered': swing_hensho_triggered,
        'swing':    [_swing_entry(r) for r in swing_top10],
        'daytrade': [_dt_entry(r)    for r in dt_top10],
    }

    # 日付別ファイル保存
    fname = data_dir / f"results_{today}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # index.json 更新（直近90日分）
    index_path = data_dir / "index.json"
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
    else:
        index = {'dates': []}
    if today not in index['dates']:
        index['dates'].insert(0, today)
    index['dates'] = sorted(set(index['dates']), reverse=True)[:90]
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"JSON保存完了: {fname}")

save_json_results()

# GitHub Pages へ自動プッシュ
import subprocess
try:
    repo = str(_HERE)
    subprocess.run(["git", "-C", repo, "add", "docs/data/"], check=True)
    result = subprocess.run(["git", "-C", repo, "diff", "--staged", "--quiet"])
    if result.returncode != 0:
        subprocess.run(["git", "-C", repo, "commit", "-m", "Update results"], check=True)
        subprocess.run(["git", "-C", repo, "pull", "--rebase", "origin", "main"], check=True)
        subprocess.run(["git", "-C", repo, "push"], check=True)
        print("GitHub Pages へプッシュ完了")
    else:
        print("変更なし: プッシュをスキップ")
except Exception as e:
    print(f"git push 失敗（メールは送信済み）: {e}")
