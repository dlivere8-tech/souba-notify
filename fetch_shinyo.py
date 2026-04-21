"""
fetch_shinyo.py
毎週金曜16:30に実行し、日経225銘柄の信用倍率をJSON保存する
souba.pyはこのJSONを参照するだけでスクレイピング不要になる
"""
import requests
import json
import time
import re
import os
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime
import pytz

_HERE = Path(__file__).parent

jst = pytz.timezone('Asia/Tokyo')
now = datetime.now(jst).strftime('%Y-%m-%d %H:%M')
print(f"信用倍率取得開始: {now}")

# ========================================
# 日経225銘柄コード取得（souba.pyと同じロジック）
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

# ========================================
# 信用倍率取得
# ========================================
def get_shinyo_bairitu(code):
    url = f"https://finance.yahoo.co.jp/quote/{code}.T"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
    }
    for attempt in range(3):
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 500:
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
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
                print(f"  エラー {code}: {e}")
    return None

# ========================================
# メイン処理
# ========================================
codes = fetch_nikkei225_codes()
shinyo_data = {}
success = 0
fail = 0

for i, code in enumerate(codes):
    ratio = get_shinyo_bairitu(code)
    if ratio is not None:
        shinyo_data[code] = ratio
        success += 1
    else:
        fail += 1

    time.sleep(0.5)
    if (i + 1) % 20 == 0:
        print(f"  進捗: {i+1}/{len(codes)} 銘柄 (取得成功: {success}, 失敗: {fail})")
        time.sleep(10)

# JSON保存
output = {
    "updated_at": now,
    "data": shinyo_data
}

with open(_HERE / "shinyo_cache.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n完了: {success}銘柄取得 / {fail}銘柄失敗")
print(f"shinyo_cache.json を保存しました")
