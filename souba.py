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
import gspread
from google.oauth2.service_account import Credentials
import json
import os

# ========================================
# 環境変数から取得（GitHubに直接書かない）
GMAIL_ADDRESS  = os.environ['GMAIL_ADDRESS']
GMAIL_APP_PASS = os.environ['GMAIL_APP_PASS']
SEND_TO        = os.environ['SEND_TO']
SHEET_ID       = os.environ['SHEET_ID']
GOOGLE_CREDS   = os.environ['GOOGLE_CREDS']
# ========================================

jst = pytz.timezone('Asia/Tokyo')
now = datetime.now(jst)
now_str = now.strftime('%Y-%m-%d %H:%M')
today   = now.strftime('%Y-%m-%d')

print(f"実行開始: {now_str}")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Referer": "https://jp.investing.com/"
}

# --- 日本10年債金利 ---
jgb_val = None
jgb_change = "-"
jgb_pct = "-"
try:
    res_inv = requests.get("https://jp.investing.com/rates-bonds/japan-10-year-bond-yield", headers=headers, timeout=8)
    soup_inv = BeautifulSoup(res_inv.text, 'html.parser')
    curr_tag   = soup_inv.find('div', {'data-test': 'instrument-price-last'})
    change_tag = soup_inv.find('span', {'data-test': 'instrument-price-change'})
    pct_tag    = soup_inv.find('span', {'data-test': 'instrument-price-change-percent'})
    if curr_tag:   jgb_val    = float(curr_tag.get_text(strip=True))
    if change_tag:
        chg = float(change_tag.get_text(strip=True))
        jgb_change = f"{chg:+.3f}"
    if pct_tag:    jgb_pct = pct_tag.get_text(strip=True).strip('()')
except Exception as e:
    print(f"JGB取得エラー: {e}")

# --- yFinanceでデータ取得 ---
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

# --- スプレッドシートに追記 ---
try:
    creds_dict = json.loads(GOOGLE_CREDS)
    creds = Credentials.from_service_account_info(creds_dict, scopes=[
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ])
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet("日次データ")
    ws.append_row([
        today,
        round(dow,    1) if dow    else "取得失敗",
        round(sp,     1) if sp     else "取得失敗",
        round(nasdaq, 1) if nasdaq else "取得失敗",
        round(usdjpy, 3) if usdjpy else "取得失敗",
        round(nikkei, 1) if nikkei else "取得失敗",
        round(growth, 1) if growth else "取得失敗",
        round(nkfut,  1) if nkfut  else "取得失敗",
        round(jgb_val,3) if jgb_val else "取得失敗",
        round(vix,    3) if vix    else "取得失敗",
        round(tnx,    3) if tnx    else "取得失敗",
        round(oil,    1) if oil    else "取得失敗",
        round(gold,   1) if gold   else "取得失敗",
    ])
    print("スプレッドシートに追記しました")
except Exception as e:
    print(f"スプレッドシートエラー: {e}")

# --- メール送信 ---
def fmt(val, is_rate=False):
    if val is None: return "取得失敗"
    return f"{val:.3f}" if is_rate else f"{val:,.1f}"

items = [
    ("ダウ平均",       fmt(dow),    dow_c,    dow_p,    False),
    ("S&P 500",        fmt(sp),     sp_c,     sp_p,     False),
    ("Nasdaq",         fmt(nasdaq), nasdaq_c, nasdaq_p, False),
    ("ドル円",         fmt(usdjpy, True), fx_c, fx_p,   True),
    ("日経平均(現物)", fmt(nikkei), nk_c,     nk_p,     False),
    ("東証グロース",   fmt(growth), gr_c,     gr_p,     False),
    ("日経先物",       fmt(nkfut),  nf_c,     nf_p,     False),
    ("日本10年債金利", f"{jgb_val:.3f}" if jgb_val else "取得失敗", jgb_change, jgb_pct, True),
    ("VIX指数",        fmt(vix, True),  vix_c, vix_p,  True),
    ("米10年債金利",   fmt(tnx, True),  tnx_c, tnx_p,  True),
    ("WTI原油",        fmt(oil),    oil_c,    oil_p,    False),
    ("金先物",         fmt(gold),   gold_c,   gold_p,   False),
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
<h2 style='margin:0;font-size:18px;'>本日の始動データ</h2>
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
</div></body></html>"""

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
