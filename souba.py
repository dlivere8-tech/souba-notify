import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

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

# 日本10年債金利
jgb_val = None
jgb_change = "-"
jgb_pct = "-"
try:
    res_inv = requests.get("https://jp.investing.com/rates-bonds/japan-10-year-bond-yield", headers=headers, timeout=8)
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
# デイトレ推奨銘柄スクリーニング
# ========================================
import pandas_ta as ta
import time

# 日経225銘柄コード
nikkei225_codes = [
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

# ========================================
# STEP 1：日経225全銘柄の売買代金を取得→上位50銘柄を選出
# ========================================
print("STEP 1：日経225全銘柄の売買代金取得中...")

volume_scores = []
for code in nikkei225_codes:
    try:
        df = yf.Ticker(f"{code}.T").history(period="5d")
        if df.empty or len(df) < 1:
            continue
        latest = df.iloc[-1]
        trading_value = latest['Volume'] * latest['Close'] / 1e8  # 億円
        volume_scores.append({
            'code': code,
            'trading_value': trading_value
        })
    except:
        continue
    time.sleep(0.1)

# 売買代金上位50銘柄を選出
volume_scores = sorted(volume_scores,
                       key=lambda x: x['trading_value'],
                       reverse=True)[:50]

# 銘柄名を取得
top50_stocks = []
for v in volume_scores:
    try:
        info = yf.Ticker(f"{v['code']}.T").info
        name = info.get('longName') or info.get('shortName') or v['code']
        # 日本語名が取れない場合は証券コードをそのまま使う
        top50_stocks.append((v['code'], name[:10]))  # 長すぎる名前は10文字に制限
    except:
        top50_stocks.append((v['code'], v['code']))
    time.sleep(0.1)

print(f"売買代金上位50銘柄を選出しました")
print(f"上位5銘柄：{[s[0] for s in top50_stocks[:5]]}")

# ========================================
# STEP 2：50銘柄のデイトレスコアを計算→TOP10を選出
# ========================================
def calc_daytrade_score(code, name):
    try:
        df = yf.Ticker(f"{code}.T").history(period="1mo")
        if df.empty or len(df) < 5:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        recent  = df.tail(20)
        latest  = df.iloc[-1]
        atr     = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        atr_pct = (atr.iloc[-1] / latest['Close'] * 100) if atr is not None else 0
        avg_range     = ((recent['High'] - recent['Low']) / recent['Close'] * 100).mean()
        trading_value = recent['Volume'].mean() * latest['Close'] / 1e8
        vol_cv        = recent['Volume'].std() / (recent['Volume'].mean() + 1e-10)
        price_score   = 1.0 if 500 <= latest['Close'] <= 15000 else 0.3
        score = (min(atr_pct/5*30, 30) + min(avg_range/5*25, 25) +
                min(trading_value/500*25, 25) + max(10-vol_cv*5, 0) +
                price_score*10)
        return {'code':code,'name':name,'price':latest['Close'],
                'atr_pct':atr_pct,'avg_range':avg_range,
                'trading_value':trading_value,'score':score}
    except:
        return None

def calc_detail(code, name, daytrade_score):
    try:
        df = yf.Ticker(f"{code}.T").history(period="1mo")
        if df.empty or len(df) < 5:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.reset_index()
        df['RSI']  = ta.rsi(df['Close'], length=14)
        df['MA25'] = df['Close'].rolling(25).mean()
        df['MA75'] = df['Close'].rolling(75).mean()
        latest = df.iloc[-1]
        curr   = latest['Close']
        rsi    = latest['RSI']
        trend  = '上昇' if latest['MA25'] > latest['MA75'] else '下落'
        pbins  = pd.cut(df['Close'], bins=10, labels=False)
        vol_by_bin  = df.groupby(pbins)['Volume'].sum()
        bin_edges   = pd.cut(df['Close'], bins=10, retbins=True)[1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        top5   = vol_by_bin.nlargest(5).index
        hvn    = sorted([bin_centers[i] for i in top5 if i < len(bin_centers)])
        sup    = max([p for p in hvn if p < curr], default=None)
        res    = min([p for p in hvn if p > curr], default=None)
        rr     = abs(res-curr)/abs(curr-sup) if sup and res and curr!=sup else 0
        return {'code':code,'name':name,'price':curr,'rsi':rsi,
                'trend':trend,'sup':sup,'res':res,'rr':rr,
                'daytrade_score':daytrade_score}
    except:
        return None

print("STEP 2：デイトレスクリーニング中...")
dt_results = []
for code, name in top50_stocks:
    r = calc_daytrade_score(code, name)
    if r:
        dt_results.append(r)
    time.sleep(0.2)

dt_results = sorted(dt_results, key=lambda x: x['score'], reverse=True)[:10]

print("STEP 3：TOP10 詳細分析中...")
detail_results = []
for r in dt_results:
    d = calc_detail(r['code'], r['name'], r['score'])
    if d:
        detail_results.append(d)
    time.sleep(0.2)

detail_results = sorted(detail_results,
                        key=lambda x: x['rr'] * x['daytrade_score'],
                        reverse=True)
print(f"デイトレスクリーニング完了：{len(detail_results)}銘柄")


# yFinanceでデータ取得
def get_yf(symbol, is_rate=False):
    try:
        df = yf.Ticker(symbol).history(period="5d")
        if df.empty or len(df) < 2:
            return None, "-", "-"
        curr = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        change = curr - prev
        pct = (change / prev) * 100
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
    ("日本10年債金利", f"{jgb_val:.3f}" if jgb_val else "取得失敗", jgb_change, jgb_pct, True),
    ("VIX指数",        fmt(vix,   True),      vix_c,    vix_p,    True),
    ("米10年債金利",   fmt(tnx,   True),      tnx_c,    tnx_p,    True),
    ("WTI原油",        fmt(oil),              oil_c,    oil_p,    False),
    ("金先物",         fmt(gold),             gold_c,   gold_p,   False),
]

rows = ""
for name, val, change, pct, is_rate in items:
    color = "#d32f2f" if str(pct).startswith('+') else "#1565c0" if str(pct).startswith('-') else "#333"
    rows += f"""<tr>
        <td style='padding:8px 12px;border-bottom:1px solid #eee;'>{name}</td>
        <td style='padding:8px 12px;border-bottom:1px solid #eee;text-align:right;font-weight:bold;'>{val}</td>
        <td style='padding:8px 12px;border-bottom:1px solid #eee;text-align:right;color:{color};'>{change}</td>
        <td style='padding:8px 12px;border-bottom:1px solid #eee;text-align:right;color:{color};font-weight:bold;'>{pct}</td>
    </tr>"""

html = f"""<html><body style='font-family:sans-serif;background:#f5f5f5;padding:20px;'>
<div style='max-width:480px;margin:0 auto;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
<div style='background:#1a237e;color:#fff;padding:16px 20px;'>
<h2 style='margin:0;font-size:18px;'>最新マーケットデータ</h2>
<p style='margin:4px 0 0;font-size:13px;opacity:0.8;'>{now_str} JST</p>
</div>
<table style='width:100%;border-collapse:collapse;font-size:14px;'>
<thead><tr style='background:#e8eaf6;'>
<th style='padding:8px 12px;text-align:left;'>指標</th>
<th style='padding:8px 12px;text-align:right;'>現在値</th>
<th style='padding:8px 12px;text-align:right;'>前日比</th>
<th style='padding:8px 12px;text-align:right;'>騰落率</th>
</tr></thead>
<tbody>{rows}</tbody></table>
<p style='padding:12px 16px;font-size:11px;color:#999;margin:0;'>投資判断はご自身の責任でお願いします</p>
dt_rows = ""
for i, d in enumerate(detail_results):
    sup_str = f"¥{d['sup']:,.0f}" if d['sup'] else "-"
    res_str = f"¥{d['res']:,.0f}" if d['res'] else "-"
    rr_str  = f"{d['rr']:.1f}" if d['rr'] > 0 else "-"
    rsi_color = "#d32f2f" if d['rsi'] < 30 else "#1565c0" if d['rsi'] > 70 else "#333"
    trend_color = "#d32f2f" if d['trend'] == '上昇' else "#1565c0"
    dt_rows += f"""<tr>
        <td style='padding:6px 12px;border-bottom:1px solid #eee;font-weight:bold;'>{i+1}. {d['name']}（{d['code']}）</td>
        <td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:right;'>¥{d['price']:,.0f}</td>
        <td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:right;color:{rsi_color};'>{d['rsi']:.1f}</td>
        <td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:right;color:{trend_color};'>{d['trend']}</td>
        <td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:right;'>{sup_str}</td>
        <td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:right;'>{res_str}</td>
        <td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:right;font-weight:bold;'>{rr_str}</td>
    </tr>"""

dt_section = f"""
<div style='margin-top:16px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
<div style='background:#1b5e20;color:#fff;padding:12px 20px;'>
<h2 style='margin:0;font-size:16px;'>今日のデイトレ推奨銘柄</h2>
<p style='margin:4px 0 0;font-size:11px;opacity:0.8;'>前日終値ベース・投資判断はご自身で</p>
</div>
<table style='width:100%;border-collapse:collapse;font-size:12px;'>
<thead><tr style='background:#e8f5e9;'>
<th style='padding:6px 12px;text-align:left;'>銘柄</th>
<th style='padding:6px 12px;text-align:right;'>株価</th>
<th style='padding:6px 12px;text-align:right;'>RSI</th>
<th style='padding:6px 12px;text-align:right;'>トレンド</th>
<th style='padding:6px 12px;text-align:right;'>損切り</th>
<th style='padding:6px 12px;text-align:right;'>利確</th>
<th style='padding:6px 12px;text-align:right;'>RR</th>
</tr></thead>
<tbody>{dt_rows}</tbody></table>
</div>"""
</div>{dt_section}</body></html>"""

msg = MIMEMultipart('alternative')
msg['From']    = GMAIL_ADDRESS
msg['To']      = SEND_TO
msg['Subject'] = f"Souba Data {now_str}"
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
    print("メール送信完了！")
except Exception as e:
    print(f"メール送信失敗: {e}")
