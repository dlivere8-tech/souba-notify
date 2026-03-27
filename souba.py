import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time
import numpy as np

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
# 【改善】日経225構成銘柄を自動取得（方法A）
# Wikipediaから取得、失敗時はフォールバックリストを使用
# ========================================
def fetch_nikkei225_codes():
    """WikipediaのNikkei 225ページから銘柄コードを取得する"""
    import re
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

    # フォールバック：ハードコードリスト
    # 【日経225銘柄リスト：入替時はここを更新】
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
    """決算日が7日以内なら「決算M/D」を返す、なければ空文字"""
    try:
        ticker = yf.Ticker(f"{code}.T")
        cal = ticker.calendar
        if cal is None:
            return ""
        if isinstance(cal, dict):
            earnings_date = cal.get('Earnings Date')
            if earnings_date is None:
                return ""
            if isinstance(earnings_date, list):
                earnings_date = earnings_date[0]
        else:
            if 'Earnings Date' not in cal.columns and 'Earnings Date' not in cal.index:
                return ""
            try:
                earnings_date = (
                    cal.loc['Earnings Date'].iloc[0]
                    if 'Earnings Date' in cal.index
                    else cal['Earnings Date'].iloc[0]
                )
            except:
                return ""
        if earnings_date is None:
            return ""
        if hasattr(earnings_date, 'tzinfo') and earnings_date.tzinfo is not None:
            earnings_date = earnings_date.replace(tzinfo=None)
        today = datetime.now().replace(tzinfo=None)
        diff = (earnings_date - today).days
        if 0 <= diff <= 7:
            return f"決算{earnings_date.month}/{earnings_date.day}"
        return ""
    except:
        return ""

# ========================================
# 【改善】銘柄名を日本語で取得
# ========================================
def get_japanese_name(code):
    """銘柄名を日本語表記で返す（取得失敗時は英語名/コード）"""
    try:
        info = yf.Ticker(f"{code}.T").info
        for key in ('displayName', 'longName', 'shortName'):
            val = info.get(key)
            if val:
                return val[:12]
        return code
    except:
        return code

# ========================================
# STEP 1：売買代金上位50銘柄を選出
# ========================================
print("STEP 1：売買代金上位50銘柄を取得中...")

volume_scores = []
for code in nikkei225_codes:
    try:
        df = yf.Ticker(f"{code}.T").history(period="5d")
        if df.empty or len(df) < 1:
            continue
        latest = df.iloc[-1]
        trading_value = latest['Volume'] * latest['Close'] / 1e8
        volume_scores.append({'code': code, 'trading_value': trading_value})
    except:
        continue
    time.sleep(0.1)

volume_scores = sorted(volume_scores, key=lambda x: x['trading_value'], reverse=True)[:50]

top50_stocks = []
for v in volume_scores:
    name = get_japanese_name(v['code'])
    top50_stocks.append((v['code'], name))
    time.sleep(0.1)

print(f"上位50銘柄選出完了")

# ========================================
# STEP 2：データ取得1回でraw値を収集
# ========================================
def calc_raw(code, name):
    """スイングのtrend/RRは後で相対評価するためraw値を返す"""
    try:
        df = yf.Ticker(f"{code}.T").history(period="3mo")
        if df.empty or len(df) < 20:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.reset_index()

        df['RSI']  = ta.rsi(df['Close'], length=14)
        df['MA25'] = df['Close'].rolling(25).mean()
        df['MA75'] = df['Close'].rolling(75).mean()
        atr        = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        recent = df.tail(20)
        latest = df.iloc[-1]
        curr   = latest['Close']
        rsi    = latest['RSI'] if not pd.isna(latest['RSI']) else 50

        # 前日比%
        change_pct = (
            (curr - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100
            if len(df) >= 2 else 0.0
        )

        # MA乖離率（プラス=上昇トレンド強い、マイナス=下落トレンド強い）
        ma25 = latest['MA25']
        ma75 = latest['MA75']
        if not pd.isna(ma25) and not pd.isna(ma75) and ma75 != 0:
            ma_divergence = (ma25 - ma75) / ma75 * 100
        else:
            ma_divergence = 0.0
        trend_up = ma_divergence > 0
        trend    = '上昇' if trend_up else '下落'

        # ----------------------------------------
        # 出来高プロファイルで節目計算
        # ----------------------------------------
        sup_buy_p = res_buy_p = sup_sell_p = res_sell_p = None
        rr_buy = rr_sell = 0.0
        try:
            bin_result  = pd.cut(df['Close'], bins=10, labels=False, retbins=True)
            pbins       = bin_result[0]
            bin_edges   = bin_result[1]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            vol_by_bin  = df.groupby(pbins)['Volume'].sum()
            top5        = vol_by_bin.nlargest(5).index
            hvn         = sorted([bin_centers[i] for i in top5 if i < len(bin_centers)])

            # 買い用：損切り=下、利確=上
            sup_buy_p = max([p for p in hvn if p < curr], default=None)
            res_buy_p = min([p for p in hvn if p > curr], default=None)
            if sup_buy_p and res_buy_p and curr != sup_buy_p:
                rr_buy = abs(res_buy_p - curr) / abs(curr - sup_buy_p)

            # 【バグ修正】売り用：損切り=上（上がったら損）、利確=下（下がったら利益）
            res_sell_p = min([p for p in hvn if p > curr], default=None)  # 損切り（上）
            sup_sell_p = max([p for p in hvn if p < curr], default=None)  # 利確（下）
            if res_sell_p and sup_sell_p and curr != res_sell_p:
                rr_sell = abs(curr - sup_sell_p) / abs(res_sell_p - curr)
        except:
            pass

        # デイトレスコア（絶対評価のまま）
        atr_val   = atr.iloc[-1] if atr is not None and not atr.isna().all() else 0
        atr_pct   = atr_val / curr * 100
        avg_range = ((recent['High'] - recent['Low']) / recent['Close'] * 100).mean()
        trade_val = recent['Volume'].mean() * curr / 1e8
        vol_cv    = recent['Volume'].std() / (recent['Volume'].mean() + 1e-10)
        price_s   = 1.0 if 500 <= curr <= 15000 else 0.3

        daytrade_score = (
            min(atr_pct / 5 * 30, 30) +
            min(avg_range / 5 * 25, 25) +
            min(trade_val / 500 * 25, 25) +
            max(10 - vol_cv * 5, 0) +
            price_s * 10
        )

        # デイトレRR有効判定（買い方向の節目で判定）
        dt_rr_valid = (sup_buy_p is not None and res_buy_p is not None and rr_buy > 0)

        # RSIスコア（絶対評価のまま）
        if 40 <= rsi <= 60:
            buy_rsi_score  = 35;  sell_rsi_score = 35
        elif 30 <= rsi < 40:
            buy_rsi_score  = 25;  sell_rsi_score = 20
        elif 60 < rsi <= 70:
            buy_rsi_score  = 20;  sell_rsi_score = 25
        elif rsi < 30:
            buy_rsi_score  = 15;  sell_rsi_score = 0   # 売られすぎ除外
        else:  # 70超
            buy_rsi_score  = 0;   sell_rsi_score = 15

        return {
            'code':           code,
            'name':           name,
            'price':          curr,
            'change_pct':     change_pct,
            'rsi':            rsi,
            'trend':          trend,
            'trend_up':       trend_up,
            'ma_divergence':  ma_divergence,
            # デイトレ
            'sup_buy':        sup_buy_p,
            'res_buy':        res_buy_p,
            'rr_buy':         rr_buy,
            'dt_rr_valid':    dt_rr_valid,
            'daytrade_score': round(daytrade_score, 1),
            # スイング raw（相対評価前）
            'rr_buy_raw':     rr_buy,
            'rr_sell_raw':    rr_sell,
            'buy_rsi_score':  buy_rsi_score,
            'sell_rsi_score': sell_rsi_score,
            # 節目価格
            'sup_buy_price':  sup_buy_p,   # 買い損切り（下）
            'res_buy_price':  res_buy_p,   # 買い利確（上）
            'res_sell_price': res_sell_p,  # 売り損切り（上）
            'sup_sell_price': sup_sell_p,  # 売り利確（下）
        }
    except Exception as e:
        print(f"  エラー {code}: {e}")
        return None

print("STEP 2：全銘柄rawデータ収集中...")
all_results = []
for code, name in top50_stocks:
    r = calc_raw(code, name)
    if r:
        all_results.append(r)
    time.sleep(0.3)

print(f"rawデータ収集完了：{len(all_results)}銘柄")

# ========================================
# 相対評価でスイングスコアを確定
# ========================================
def rank_score(values, higher_is_better=True, max_pts=35):
    """順位ベースで0〜max_ptsの点数リストを返す（同値は平均順位、numpy実装）"""
    n = len(values)
    if n <= 1:
        return [max_pts] * n
    arr = np.array(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # 同値グループを平均順位に置き換え
    i = 0
    while i < n:
        j = i
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    if higher_is_better:
        return [(r - 1) / (n - 1) * max_pts for r in ranks]
    else:
        return [(n - r) / (n - 1) * max_pts for r in ranks]

ma_divs  = [r['ma_divergence'] for r in all_results]
rr_buys  = [r['rr_buy_raw']    for r in all_results]
rr_sells = [r['rr_sell_raw']   for r in all_results]

# 買いトレンド：MA乖離率が大きい（上昇強い）ほど高得点
buy_trend_sc  = rank_score(ma_divs,  higher_is_better=True,  max_pts=35)
# 売りトレンド：MA乖離率が小さい（下落強い）ほど高得点
sell_trend_sc = rank_score(ma_divs,  higher_is_better=False, max_pts=35)
# RR比率：大きいほど高得点（買い・売り独立）
buy_rr_sc     = rank_score(rr_buys,  higher_is_better=True,  max_pts=30)
sell_rr_sc    = rank_score(rr_sells, higher_is_better=True,  max_pts=30)

for i, r in enumerate(all_results):
    buy_score  = buy_trend_sc[i]  + r['buy_rsi_score']  + buy_rr_sc[i]
    sell_score = sell_trend_sc[i] + r['sell_rsi_score'] + sell_rr_sc[i]

    buy_rr_valid = (
        r['sup_buy_price'] is not None and
        r['res_buy_price'] is not None and
        r['rr_buy_raw'] > 0
    )
    # 【バグ修正】売りRR有効判定は売り専用の節目で判定
    sell_rr_valid = (
        r['res_sell_price'] is not None and
        r['sup_sell_price'] is not None and
        r['rr_sell_raw'] > 0
    )

    if buy_score >= sell_score:
        r['swing_direction'] = "買い"
        r['swing_score']     = round(buy_score, 1)
        r['swing_sup']       = r['sup_buy_price']   # 損切り（下）
        r['swing_res']       = r['res_buy_price']   # 利確（上）
        r['swing_rr']        = r['rr_buy_raw']
        r['swing_rr_valid']  = buy_rr_valid
    else:
        r['swing_direction'] = "売り"
        r['swing_score']     = round(sell_score, 1)
        r['swing_sup']       = r['res_sell_price']  # 【修正】損切り（上）
        r['swing_res']       = r['sup_sell_price']  # 【修正】利確（下）
        r['swing_rr']        = r['rr_sell_raw']
        r['swing_rr_valid']  = sell_rr_valid

print("スイングスコア相対評価完了")

# 決算日フラグ取得
print("STEP 3：決算日フラグ取得中...")
earnings_flags = {}
for code, name in top50_stocks:
    flag = get_earnings_flag(code)
    if flag:
        earnings_flags[code] = flag
        print(f"  {code} {name}: ⚠{flag}")
    time.sleep(0.2)

# ========================================
# RR算出不可を除外してTOP10選出
# ========================================
dt_valid    = [r for r in all_results if r['dt_rr_valid']]
dt_top10    = sorted(dt_valid,    key=lambda x: x['daytrade_score'], reverse=True)[:10]

swing_valid = [r for r in all_results if r['swing_rr_valid']]
swing_top10 = sorted(swing_valid, key=lambda x: x['swing_score'],    reverse=True)[:10]

print(f"デイトレ対象：{len(dt_valid)}銘柄 / スイング対象：{len(swing_valid)}銘柄")

# ========================================
# デイトレ/スイング目線相違フラグ
# ========================================
dt_dir_map    = {r['code']: ("買い" if r['change_pct'] >= 0 else "売り") for r in dt_top10}
swing_dir_map = {r['code']: r['swing_direction'] for r in swing_top10}

MISMATCH_CODES = {
    code for code in set(dt_dir_map) & set(swing_dir_map)
    if dt_dir_map[code] != swing_dir_map[code]
}
if MISMATCH_CODES:
    print(f"目線相違銘柄: {MISMATCH_CODES}")

# ========================================
# yFinanceで相場データ取得
# ========================================
def get_yf(symbol, is_rate=False):
    try:
        df = yf.Ticker(symbol).history(period="5d")
        if df.empty or len(df) < 2:
            return None, "-", "-"
        curr   = df['Close'].iloc[-1]
        prev   = df['Close'].iloc[-2]
        change = curr - prev
        pct    = (change / prev) * 100
        if is_rate:
            return curr, f"{change:+.3f}", f"{pct:+.2f}%"
        else:
            return curr, f"{change:+,.1f}", f"{pct:+.2f}%"
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
    ("ダウ平均",       fmt(dow),              dow_c,    dow_p,    False),
    ("S&P 500",        fmt(sp),               sp_c,     sp_p,     False),
    ("Nasdaq",         fmt(nasdaq),           nasdaq_c, nasdaq_p, False),
    ("ドル円",         fmt(usdjpy, True),     fx_c,     fx_p,     True),
    ("日経平均(現物)", fmt(nikkei),           nk_c,     nk_p,     False),
    ("東証グロース",   fmt(growth),           gr_c,     gr_p,     False),
    ("日経先物",       fmt(nkfut),            nf_c,     nf_p,     False),
    ("日本10年債金利",
     f"{jgb_val:.3f}" if jgb_val else "取得失敗",
     jgb_change, jgb_pct, True),
    ("VIX指数",        fmt(vix,   True),      vix_c,    vix_p,    True),
    ("米10年債金利",   fmt(tnx,   True),      tnx_c,    tnx_p,    True),
    ("WTI原油",        fmt(oil),              oil_c,    oil_p,    False),
    ("金先物",         fmt(gold),             gold_c,   gold_p,   False),
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
    """デイトレ推奨表（目線列なし・日本語銘柄名）"""
    if mismatch_codes is None:
        mismatch_codes = set()
    # 【改善】目線列を削除
    thead = (
        "<tr style='background:#f5f5f5;'>"
        "<th style='padding:6px 8px;text-align:left;font-size:11px;'>銘柄</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>株価</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>前日比</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>RSI</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>損切り</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>利確</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>RR</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>スコア</th>"
        "</tr>"
    )
    tbody = ""
    for i, d in enumerate(results):
        earnings = earnings_flags.get(d['code'], "")
        earnings_html = (
            " <span style='color:#e65100;font-size:10px;font-weight:bold;'>"
            + "\u26a0" + earnings + "</span>"
        ) if earnings else ""

        mismatch_html = (
            " <span style='color:#6a1b9a;font-size:10px;font-weight:bold;'>"
            + "\u26a0" + "目線相違" + "</span>"
        ) if d['code'] in mismatch_codes else ""

        pct_val   = d['change_pct']
        pct_str   = f"{pct_val:+.2f}%"
        pct_color = "#d32f2f" if pct_val >= 0 else "#1565c0"

        sup_str = f"{d['sup_buy']:,.0f}" + "円" if d['sup_buy'] else "-"
        res_str = f"{d['res_buy']:,.0f}" + "円" if d['res_buy'] else "-"
        rr_str  = f"{d['rr_buy']:.1f}"  if d['rr_buy'] > 0 else "-"

        rsi_color = "#d32f2f" if d['rsi'] < 30 else "#1565c0" if d['rsi'] > 70 else "#333"

        tbody += (
            "<tr>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;font-weight:bold;font-size:11px;'>"
            + str(i+1) + ". " + d['name'] + "(" + d['code'] + ")"
            + earnings_html + mismatch_html + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-size:11px;'>"
            + f"{d['price']:,.0f}" + "円</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-size:11px;color:"
            + pct_color + ";font-weight:bold;'>" + pct_str + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;color:"
            + rsi_color + ";font-size:11px;'>" + f"{d['rsi']:.1f}" + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-size:11px;'>"
            + sup_str + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-size:11px;'>"
            + res_str + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-weight:bold;font-size:11px;'>"
            + rr_str + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-weight:bold;font-size:12px;color:#1b5e20;'>"
            + str(d['daytrade_score']) + "</td>"
            + "</tr>"
        )
    return (
        "<div style='margin-top:16px;background:#fff;border-radius:8px;"
        "overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>"
        "<div style='background:#1b5e20;color:#fff;padding:12px 16px;'>"
        "<h2 style='margin:0;font-size:15px;'>今日のデイトレ推奨銘柄</h2>"
        "<p style='margin:4px 0 0;font-size:11px;opacity:0.8;'>"
        "前日終値ベース・RR算出可能銘柄のみ・投資判断はご自身で</p>"
        "</div>"
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead>" + thead + "</thead>"
        "<tbody>" + tbody + "</tbody>"
        "</table></div></div>"
    )


def build_swing_table(results, earnings_flags, mismatch_codes=None):
    """スイング推奨表（買い/売り両対応・損切り利確反転修正済み）"""
    if mismatch_codes is None:
        mismatch_codes = set()
    thead = (
        "<tr style='background:#f5f5f5;'>"
        "<th style='padding:6px 8px;text-align:left;font-size:11px;'>銘柄</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>株価</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>前日比</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>方向</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>RSI</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>トレンド</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>損切り</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>利確</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>RR</th>"
        "<th style='padding:6px 8px;text-align:right;font-size:11px;'>スコア</th>"
        "</tr>"
    )
    tbody = ""
    for i, d in enumerate(results):
        earnings = earnings_flags.get(d['code'], "")
        earnings_html = (
            " <span style='color:#e65100;font-size:10px;font-weight:bold;'>"
            + "\u26a0" + earnings + "</span>"
        ) if earnings else ""

        mismatch_html = (
            " <span style='color:#6a1b9a;font-size:10px;font-weight:bold;'>"
            + "\u26a0" + "目線相違" + "</span>"
        ) if d['code'] in mismatch_codes else ""

        pct_val   = d['change_pct']
        pct_str   = f"{pct_val:+.2f}%"
        pct_color = "#d32f2f" if pct_val >= 0 else "#1565c0"

        direction = d['swing_direction']
        if direction == "買い":
            dir_color   = "#d32f2f"
            dir_label   = "↑買い"
            score_color = "#b71c1c"
        else:
            dir_color   = "#1565c0"
            dir_label   = "↓売り"
            score_color = "#0d47a1"

        # swing_sup=損切り、swing_res=利確（売り時は反転済み）
        sup_str = f"{d['swing_sup']:,.0f}" + "円" if d['swing_sup'] else "-"
        res_str = f"{d['swing_res']:,.0f}" + "円" if d['swing_res'] else "-"
        rr_str  = f"{d['swing_rr']:.1f}"  if d['swing_rr'] > 0 else "-"

        rsi_color   = "#d32f2f" if d['rsi'] < 30 else "#1565c0" if d['rsi'] > 70 else "#333"
        trend_color = "#d32f2f" if d['trend'] == '上昇' else "#1565c0"

        tbody += (
            "<tr>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;font-weight:bold;font-size:11px;'>"
            + str(i+1) + ". " + d['name'] + "(" + d['code'] + ")"
            + earnings_html + mismatch_html + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-size:11px;'>"
            + f"{d['price']:,.0f}" + "円</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-size:11px;color:"
            + pct_color + ";font-weight:bold;'>" + pct_str + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-size:11px;color:"
            + dir_color + ";font-weight:bold;'>" + dir_label + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;color:"
            + rsi_color + ";font-size:11px;'>" + f"{d['rsi']:.1f}" + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;color:"
            + trend_color + ";font-size:11px;'>" + d['trend'] + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-size:11px;'>"
            + sup_str + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-size:11px;'>"
            + res_str + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-weight:bold;font-size:11px;'>"
            + rr_str + "</td>"
            + "<td style='padding:6px 8px;border-bottom:1px solid #eee;text-align:right;font-weight:bold;font-size:12px;color:"
            + score_color + ";'>" + str(d['swing_score']) + "</td>"
            + "</tr>"
        )
    return (
        "<div style='margin-top:16px;background:#fff;border-radius:8px;"
        "overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>"
        "<div style='background:#1a237e;color:#fff;padding:12px 16px;'>"
        "<h2 style='margin:0;font-size:15px;'>今週のスイング推奨銘柄</h2>"
        "<p style='margin:4px 0 0;font-size:11px;opacity:0.8;'>"
        "買い/売り両対応・RR算出可能銘柄のみ・スコア上位10</p>"
        "</div>"
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead>" + thead + "</thead>"
        "<tbody>" + tbody + "</tbody>"
        "</table></div></div>"
    )


dt_section    = build_daytrade_table(dt_top10,    earnings_flags, MISMATCH_CODES)
swing_section = build_swing_table(swing_top10, earnings_flags, MISMATCH_CODES)

html = (
    "<html><body style='font-family:sans-serif;background:#f5f5f5;padding:20px;'>"
    "<div style='max-width:600px;margin:0 auto;'>"
    "<div style='background:#fff;border-radius:8px;overflow:hidden;"
    "box-shadow:0 2px 8px rgba(0,0,0,0.1);'>"
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
    "<p style='padding:12px 16px;font-size:11px;color:#999;margin:0;'>"
    "投資判断はご自身の責任でお願いします</p>"
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
