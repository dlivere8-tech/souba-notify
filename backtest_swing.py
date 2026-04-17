"""
スイング推奨バックテスト（本格版）

エントリー条件：
  推奨日の翌営業日の寄り付き価格でエントリー
  買い推奨→買いエントリー、売り推奨→売りエントリー

決済条件（保有上限5日）：
  利確：エントリーから5日以内に利確ライン（HVNベース）に届いたら勝ち
  損切：エントリーから5日以内に損切りライン（ATRベース）を踏んだら負け
  同日に両方触れた場合→損切り扱い
  5日以内にどちらも触れない→5日目の引け値で強制決済し損益計算

出力指標：
  勝率・プロフィットファクター・平均RR
  MFE（最大有利展開）・MAE（最大不利展開）
  従来の「5日後方向的中率」との比較も併記

注意：信用倍率取得はバックテスト時スキップ
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import csv
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# ユニバース（日経225）
# ─────────────────────────────────────────────
UNIVERSE = [
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
    "9616","9735","9766","9983","9984",
]

HOLD_DAYS       = 5
TOP_N           = 10
TOP_VOL         = 50
HIST_BARS       = 80
SL_BUFFER_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]  # SL距離拡張倍率（ATR×N倍）

# ─────────────────────────────────────────────
# 指値バックテスト パラメータ
# ─────────────────────────────────────────────
RR_TARGETS      = [1.5, 2.0, 2.5, 3.0]  # テストするRR目標
ENTRY_WINDOW    = 3      # エントリー待機日数（シグナル翌日から最大N日）
SL_BUF          = 0.003  # SL: サポートの0.3%下まで粘る
TP_BUF          = 0.005  # TP: レジスタンスの0.5%手前で逃げる
ATR_FILTER_MULT = 2.0    # エントリー価格がsignal_closeからN×ATR以内のみ採用

# ─────────────────────────────────────────────
# テクニカル指標
# ─────────────────────────────────────────────

def _rsi(closes, period=14):
    arr = np.asarray(closes, dtype=float)
    if len(arr) < period + 2:
        return 50.0
    deltas = np.diff(arr)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = gains[:period].mean()
    avg_l  = losses[:period].mean()
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i])  / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    return 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)


def _atr(highs, lows, closes, period=14):
    h, l, c = np.asarray(highs, float), np.asarray(lows, float), np.asarray(closes, float)
    if len(h) < period + 1:
        return 0.0
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = tr[:period].mean()
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
    return float(atr)


def _bb_pctb(closes, period=20, nstd=2):
    arr = np.asarray(closes, float)
    if len(arr) < period:
        return None
    sl   = arr[-period:]
    mean = sl.mean()
    std  = sl.std(ddof=0)
    if std == 0:
        return None
    upper = mean + nstd * std
    lower = mean - nstd * std
    return float((arr[-1] - lower) / (upper - lower)) if upper != lower else None


def _ema(arr, period):
    arr = np.asarray(arr, float)
    if len(arr) < period:
        return np.array([])
    k   = 2.0 / (period + 1)
    out = [arr[:period].mean()]
    for v in arr[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return np.array(out)


def _macd_lines(closes):
    arr = np.asarray(closes, float)
    if len(arr) < 34:
        return np.array([]), np.array([])
    e12 = _ema(arr, 12)
    e26 = _ema(arr, 26)
    macd_full = e26 - e12[14:]
    if len(macd_full) < 9:
        return macd_full, np.array([])
    signal = _ema(macd_full, 9)
    return macd_full[8:], signal


def calc_support_resistance(df, curr, direction, atr_val):
    """
    過去データから価格帯別出来高×反転回数でサポレジを検出（souba.pyと同一ロジック）。

    direction: 'buy' or 'sell'
    戻り値: sl_price, tp_price, sl_strength, tp_strength
    """
    try:
        if len(df) < 60:
            return None, None, 0, 0

        close = df['Close'].values.astype(float)
        high  = df['High'].values.astype(float)
        low   = df['Low'].values.astype(float)
        vol   = df['Volume'].values.astype(float)
        n     = len(df)

        # ①価格帯別出来高（bins=20・直近3ヶ月×2重み）
        price_min = close.min() * 0.98
        price_max = close.max() * 1.02
        bins = 20
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        cutoff = max(0, n - 63)
        weights = np.ones(n)
        weights[cutoff:] = 2.0

        vol_by_bin = np.zeros(bins)
        for i in range(n):
            idx = np.searchsorted(bin_edges[1:], close[i])
            idx = min(idx, bins - 1)
            vol_by_bin[idx] += vol[i] * weights[i]

        # ②反転回数カウント
        reversal_by_bin = np.zeros(bins)
        for i in range(1, n - 1):
            if high[i] > high[i-1] and high[i] > high[i+1]:
                idx = np.searchsorted(bin_edges[1:], high[i])
                idx = min(idx, bins - 1)
                reversal_by_bin[idx] += 1
            if low[i] < low[i-1] and low[i] < low[i+1]:
                idx = np.searchsorted(bin_edges[1:], low[i])
                idx = min(idx, bins - 1)
                reversal_by_bin[idx] += 1

        # ③複合スコア（出来高×反転回数）
        vol_norm      = vol_by_bin / (vol_by_bin.max() + 1e-10)
        reversal_norm = reversal_by_bin / (reversal_by_bin.max() + 1e-10)
        composite     = vol_norm * 0.6 + reversal_norm * 0.4

        # ④現在値の上下に分けてTop候補を選ぶ
        curr_bin = np.searchsorted(bin_edges[1:], curr)
        curr_bin = min(curr_bin, bins - 1)

        # ATRベースの最小距離制約
        # SL: 0.5ATR以上離れたサポートのみ
        # TP: 1.5ATR以上離れたレジスタンスのみ（RR≥1.5を構造的に担保）
        min_sl_dist = atr_val * 0.5
        min_tp_dist = atr_val * 1.5

        support_bins = [(i, composite[i], bin_centers[i])
                        for i in range(curr_bin)
                        if bin_centers[i] < curr - min_sl_dist]
        support_bins.sort(key=lambda x: x[1], reverse=True)

        resist_bins = [(i, composite[i], bin_centers[i])
                       for i in range(curr_bin + 1, bins)
                       if bin_centers[i] > curr + min_tp_dist]
        resist_bins.sort(key=lambda x: x[1], reverse=True)

        def strength(score):
            if score >= 0.7: return 3
            if score >= 0.4: return 2
            return 1

        if direction == 'buy':
            sl_price    = support_bins[0][2]  if support_bins else curr - atr_val * 1.5
            sl_strength = strength(support_bins[0][1]) if support_bins else 1
            tp_price    = resist_bins[0][2]   if resist_bins  else curr + atr_val * 3.0
            tp_strength = strength(resist_bins[0][1]) if resist_bins else 1
        else:
            sl_price    = resist_bins[0][2]   if resist_bins  else curr + atr_val * 1.5
            sl_strength = strength(resist_bins[0][1]) if resist_bins else 1
            tp_price    = support_bins[0][2]  if support_bins else curr - atr_val * 3.0
            tp_strength = strength(support_bins[0][1]) if support_bins else 1

        return sl_price, tp_price, sl_strength, tp_strength

    except Exception:
        return None, None, 0, 0


def calc_entry_params(sl_zone, tp_zone, rr_target, direction,
                      sl_buf=SL_BUF, tp_buf=TP_BUF):
    """
    サポレジゾーンとRR目標から SL・TP・エントリー価格を逆算。

    買い:
      sl_actual = support  × (1 - sl_buf)   ← サポートの少し下
      tp_actual = resist   × (1 - tp_buf)   ← 抵抗線の少し手前
      entry     = (tp_actual + rr × sl_actual) / (1 + rr)  ← SL寄りに位置する

    売り:
      sl_actual = resist   × (1 + sl_buf)   ← 抵抗線の少し上
      tp_actual = support  × (1 + tp_buf)   ← サポートの少し手前（上側）
      entry     = (tp_actual + rr × sl_actual) / (1 + rr)  ← 同じ式で成立

    戻り値: (sl_actual, tp_actual, entry_calc)
    """
    if direction == '買い':
        sl_actual = sl_zone * (1 - sl_buf)
        tp_actual = tp_zone * (1 - tp_buf)
    else:
        sl_actual = sl_zone * (1 + sl_buf)
        tp_actual = tp_zone * (1 + tp_buf)
    entry_calc = (tp_actual + rr_target * sl_actual) / (1 + rr_target)
    return sl_actual, tp_actual, entry_calc


def calc_macd_cross(closes, direction):
    try:
        ml, sl = _macd_lines(np.asarray(closes, float))
        n = min(len(ml), len(sl))
        if n < 4:
            return 0
        for i in range(n - 3, n):
            prev_diff = ml[i-1] - sl[i-1]
            curr_diff = ml[i]   - sl[i]
            if direction == "買い" and prev_diff < 0 and curr_diff >= 0: return 10
            if direction == "売り" and prev_diff > 0 and curr_diff <= 0: return 10
    except Exception:
        pass
    return 0


def calc_indicators(df):
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.reset_index().rename(columns={'Date': 'Date', 'index': 'Date'})
        if 'Date' not in df.columns:
            df = df.reset_index()
            df.columns = ['Date'] + list(df.columns[1:])

        n = len(df)
        if n < HIST_BARS:
            return None

        closes  = df['Close'].values.astype(float)
        highs   = df['High'].values.astype(float)
        lows    = df['Low'].values.astype(float)
        volumes = df['Volume'].values.astype(float)

        curr = closes[-1]
        prev = closes[-2]
        change_pct = (curr - prev) / prev * 100 if prev else 0.0

        rsi = _rsi(closes)

        ma5  = float(np.mean(closes[-5:]))  if n >=  5 else float('nan')
        ma25 = float(np.mean(closes[-25:])) if n >= 25 else float('nan')
        ma75 = float(np.mean(closes[-75:])) if n >= 75 else float('nan')
        if np.isnan(ma75) or ma75 == 0:
            ma_divergence = 0.0
        else:
            ma_divergence = (ma25 - ma75) / ma75 * 100
        # MA5/25 短期トレンド（デュアルMAフィルター用）
        ma5_25_bull = (not np.isnan(ma5) and not np.isnan(ma25) and ma5 >= ma25)

        atr_val = _atr(highs, lows, closes)
        pct_b   = _bb_pctb(closes)

        # calc_support_resistance でサポレジ計算（souba.pyと同一ロジック）
        buy_sl,  buy_tp,  _, _ = calc_support_resistance(df, curr, 'buy',  atr_val)
        sell_sl, sell_tp, _, _ = calc_support_resistance(df, curr, 'sell', atr_val)

        # ATRベース実用損切り（エントリー価格 ± 1ATR）
        atr_sl_buy  = curr - atr_val   # 買い: -1ATR
        atr_sl_sell = curr + atr_val   # 売り: +1ATR

        # RR計算：ATRベース（実用的RR = TP距離 / 1ATR）
        atr_rr_buy  = abs(buy_tp  - curr) / atr_val if atr_val > 0 and buy_tp  is not None else 0.0
        atr_rr_sell = abs(curr - sell_tp) / atr_val if atr_val > 0 and sell_tp is not None else 0.0

        return {
            'curr':          curr,
            'change_pct':    change_pct,
            'rsi':           rsi,
            'ma_divergence': ma_divergence,
            'ma5_25_bull':   ma5_25_bull,   # MA5>MA25(短期↑)
            'atr_val':       atr_val,
            'pct_b':         pct_b,
            'rr_buy_raw':    atr_rr_buy,    # ATRベースRR（実用値）
            'rr_sell_raw':   atr_rr_sell,   # ATRベースRR（実用値）
            'sup_price':     sell_tp,       # 売り利確（HVN）
            'res_price':     buy_tp,        # 買い利確（HVN）
            'buy_sl_price':  atr_sl_buy,    # 買い損切（ATR×1.0）
            'sell_sl_price': atr_sl_sell,   # 売り損切（ATR×1.0）
            'hvn_sup':       buy_sl,        # HVN下値めど（参考）
            'hvn_res':       sell_sl,       # HVN上値めど（参考）
            '_closes':       closes,
            '_df':           df,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────
# RSIスコアテーブル
# ─────────────────────────────────────────────

def rsi_score_new(rsi):
    if   40 <= rsi <= 60: return 35, 35
    elif 30 <= rsi <  40: return 28, 14
    elif 60 <  rsi <= 70: return 14, 28
    elif rsi <  30:       return 35,  0
    else:                 return  0, 35


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


# ─────────────────────────────────────────────
# スイングスコア計算（NEW配点）
# ─────────────────────────────────────────────

def score_swing_new(raws):
    """NEW: トレンド20(絶対) + RSI35 + %B20 + RR15(絶対) + MACD10
    【改善】逆トレンド方向シグナルに-20ptペナルティ
    　　　　（上昇中の売り・下降中の買いを上位から排除）
    """
    for r in raws:
        # 乖離率2%未満は「方向性なし」としてトレンドスコア0pt
        div_abs = abs(r['ma_divergence'])
        if div_abs < 2.0:
            b_tr, s_tr = 0, 0
        elif r['ma_divergence'] > 0:
            b_tr, s_tr = 20, 0
        else:
            b_tr, s_tr = 0, 20

        b_rsi, s_rsi = rsi_score_new(r['rsi'])
        pctb = r['pct_b']
        b_bb  = max(0, round((1 - pctb) * 20, 1)) if pctb is not None else 0
        s_bb  = max(0, round(pctb       * 20, 1)) if pctb is not None else 0

        buy_tmp  = b_tr + b_rsi + rr_score_abs(r['rr_buy_raw'])  + b_bb
        sell_tmp = s_tr + s_rsi + rr_score_abs(r['rr_sell_raw']) + s_bb
        tent = "買い" if buy_tmp >= sell_tmp else "売り"

        macd = calc_macd_cross(r['_closes'], tent)
        buy_score  = buy_tmp  + (macd if tent == "買い" else 0)
        sell_score = sell_tmp + (macd if tent == "売り" else 0)

        # 逆トレンド方向ペナルティ：上昇中の売り・下降中の買いは-20pt
        if   tent == "売り" and r['ma_divergence'] > 0:
            sell_score = max(0, round(sell_score - 20, 1))
        elif tent == "買い" and r['ma_divergence'] < 0:
            buy_score  = max(0, round(buy_score  - 20, 1))

        atr = r['atr_val']
        if buy_score >= sell_score:
            r['swing_direction'] = "買い"
            r['swing_score']     = round(buy_score, 1)
            r['swing_tp']        = r['res_price']        # 利確 = 遠HVN上値
            r['swing_sl']        = r['buy_sl_price']     # 損切 = 近HVN下値
            r['swing_rr']        = r['rr_buy_raw']
        else:
            r['swing_direction'] = "売り"
            r['swing_score']     = round(sell_score, 1)
            r['swing_tp']        = r['sup_price']        # 利確 = 遠HVN下値
            r['swing_sl']        = r['sell_sl_price']    # 損切 = 近HVN上値
            r['swing_rr']        = r['rr_sell_raw']
        r['swing_rr_valid'] = (r['swing_tp'] is not None and r['swing_sl'] is not None)


# ─────────────────────────────────────────────
# 除外ルール（ルール2のみ：前日比±3%超除外）
# ─────────────────────────────────────────────

MIN_DIVERGENCE = 2.0   # MA乖離率の最低ライン（絶対値）。2%未満は「方向性不明」として除外

def apply_exclusions(raws, rule2_threshold=3.0, stock_dual_ma=False, min_rr=0.0):
    """
    stock_dual_ma=True: 個別銘柄のMA5/25 AND MA25/75 が両方シグナル方向と一致するものだけ通過
    min_rr: RR最低閾値（0.0=制限なし）
    """
    result = [
        r for r in raws
        if abs(r['change_pct']) <= rule2_threshold
        and abs(r.get('ma_divergence', 0)) >= MIN_DIVERGENCE
        and r.get('swing_rr', 0) >= min_rr
    ]
    if stock_dual_ma:
        filtered = []
        for r in result:
            direction  = r.get('swing_direction')
            ma25_75_ok = (r.get('ma_divergence', 0) > 0) if direction == '買い' else (r.get('ma_divergence', 0) < 0)
            ma5_25_ok  = r.get('ma5_25_bull', False)     if direction == '買い' else (not r.get('ma5_25_bull', True))
            if ma25_75_ok and ma5_25_ok:
                filtered.append(r)
        return filtered
    return result


# ─────────────────────────────────────────────
# バックテスト本体（本格版）
# ─────────────────────────────────────────────

def run_backtest(all_data_dict, eval_dates, sl_buffer_mult=1.0,
                 nikkei_data=None, use_market_filter=False,
                 nikkei_fast=25, nikkei_slow=75,
                 dual_ma_filter=False, stock_dual_ma=False,
                 hybrid_ma=False, min_rr=0.0,
                 rr_target_sl=None):
    """
    本格版バックテスト：翌日寄り付きエントリー、TP/SL判定、強制決済
    sl_buffer_mult:      ATRベースSL倍率（1.0=1ATR, 2.0=2ATR ...）
    nikkei_data:         日経平均DataFrameまたはNone
    use_market_filter:   Trueの場合、日経トレンドと逆方向シグナルを除外
    nikkei_fast/slow:    トレンド判定に使うMAの期間
    stock_dual_ma:       Trueなら個別銘柄のMA5/25 AND MA25/75 両方一致のみ通過
    dual_ma_filter:      TrueならMA5/25 AND MA25/75の両方が一致する日のみ通過
    hybrid_ma:           True=日経デュアルMA一致時は日経方向フィルター、
                         転換期は銘柄デュアルMAフィルターにフォールバック
    min_rr:              RR最低閾値（0.0=制限なし）
    rr_target_sl:        目標RRからSLを逆算（例:2.0→SL=TP距離÷2.0）。
                         Noneの場合はsl_buffer_mult×ATRを使用
    """
    trade_records = []   # 全トレード詳細
    direction_records = []  # 従来の方向的中率用

    for eval_dt in eval_dates:
        eval_ts = pd.Timestamp(eval_dt).normalize()

        # --- 売買代金でTop50に絞る ---
        slices = {}
        for code, df_full in all_data_dict.items():
            df_slice = df_full[df_full.index.normalize() < eval_ts]
            if len(df_slice) >= HIST_BARS:
                slices[code] = df_slice

        vol_scores = []
        for code, df_s in slices.items():
            recent5 = df_s.tail(5)
            trade_val = (recent5['Volume'] * recent5['Close']).mean() / 1e8
            vol_scores.append((code, trade_val))
        vol_scores.sort(key=lambda x: x[1], reverse=True)
        top_codes = {code for code, _ in vol_scores[:TOP_VOL]}

        raws = []
        for code in top_codes:
            raw = calc_indicators(slices[code])
            if raw is None:
                continue
            raw['code'] = code
            raws.append(raw)

        if len(raws) < 3:
            continue

        import copy
        raws_copy = copy.deepcopy(raws)
        score_swing_new(raws_copy)

        candidates = [r for r in raws_copy if r.get('swing_rr_valid', False)]
        # hybrid_maの場合は銘柄デュアルMAを後で条件分岐するのでここでは適用しない
        candidates = apply_exclusions(candidates, stock_dual_ma=(stock_dual_ma and not hybrid_ma), min_rr=min_rr)
        if not candidates:
            continue

        # 市場環境フィルター：日経トレンドと逆方向シグナルを除外
        if (use_market_filter or hybrid_ma) and nikkei_data is not None:
            if hybrid_ma:
                # ハイブリッドMAフィルター：
                #   日経MA5/25 AND MA25/75 一致 → 日経方向フィルター
                #   転換期（不一致）→ 銘柄デュアルMA（MA5/25 AND MA25/75両方一致）でフィルター
                trend_short = calc_nikkei_trend(nikkei_data, eval_ts, fast=5,  slow=25)
                trend_long  = calc_nikkei_trend(nikkei_data, eval_ts, fast=25, slow=75)
                if trend_short is not None and trend_long is not None:
                    nikkei_dir = trend_short if trend_short == trend_long else None
                    filtered = []
                    for r in candidates:
                        direction  = r.get('swing_direction')
                        ma25_75_ok = (r.get('ma_divergence', 0) > 0) if direction == '買い' \
                                     else (r.get('ma_divergence', 0) < 0)
                        ma5_25_ok  = r.get('ma5_25_bull', False) if direction == '買い' \
                                     else (not r.get('ma5_25_bull', True))
                        stock_ok   = ma25_75_ok and ma5_25_ok
                        if nikkei_dir is None:
                            # 転換期：MA5/25のみで判定（MA25/75不要）
                            # 理由：回復途上の銘柄はMA5>MA25だがMA25<MA75→dual要求すると買い全滅
                            if ma5_25_ok:
                                filtered.append(r)
                        elif direction == nikkei_dir:
                            # 順張り：無条件通過
                            filtered.append(r)
                        elif stock_ok:
                            # 逆張り：個別デュアルMAが揃えば通過（-10ptペナルティは別途）
                            filtered.append(r)
                    candidates = filtered
            elif dual_ma_filter:
                # デュアルMAフィルター：MA5/25 AND MA25/75 両方が一致する方向のみ通過
                trend_short = calc_nikkei_trend(nikkei_data, eval_ts, fast=5,  slow=25)
                trend_long  = calc_nikkei_trend(nikkei_data, eval_ts, fast=25, slow=75)
                if trend_short is not None and trend_long is not None:
                    if trend_short == trend_long:
                        # 両方一致 → その方向のシグナルのみ通過
                        candidates = [r for r in candidates
                                      if r['swing_direction'] == trend_short]
                    else:
                        # 不一致（転換期）→ 全シグナル除外
                        candidates = []
            else:
                nikkei_trend = calc_nikkei_trend(nikkei_data, eval_ts,
                                                 fast=nikkei_fast, slow=nikkei_slow)
                if nikkei_trend is not None:
                    candidates = [r for r in candidates
                                  if r['swing_direction'] == nikkei_trend]
            if not candidates:
                continue

        top = sorted(candidates, key=lambda x: x['swing_score'], reverse=True)[:TOP_N]

        for rank, s in enumerate(top, 1):
            code     = s['code']
            df_full  = all_data_dict[code]
            direction    = s['swing_direction']
            tp_level     = s['swing_tp']
            sl_level     = s['swing_sl']
            ma_div       = s['ma_divergence']
            trend_aligned = (
                (direction == "買い" and ma_div > 0) or
                (direction == "売り" and ma_div < 0)
            )

            # 翌営業日以降の棒足を取得
            future_bars = df_full[df_full.index.normalize() > eval_ts]
            if len(future_bars) < 1:
                continue

            # エントリー = 翌営業日の寄り付き
            entry_bar   = future_bars.iloc[0]
            entry_price = float(entry_bar['Open'])

            # SL計算：RR逆算方式 or ATRバッファ方式
            atr_val = s.get('atr_val', 0)
            if rr_target_sl is not None and tp_level is not None and rr_target_sl > 0:
                # RR逆算：SL = エントリー ± (TP距離 ÷ 目標RR)
                tp_dist = abs(tp_level - entry_price)
                sl_dist = tp_dist / rr_target_sl
                if direction == "買い":
                    sl_level = entry_price - sl_dist
                else:
                    sl_level = entry_price + sl_dist
            elif atr_val > 0:
                # ATRバッファ：SL = エントリー ± (ATR × 倍率)
                if direction == "買い":
                    sl_level = entry_price - atr_val * sl_buffer_mult
                else:
                    sl_level = entry_price + atr_val * sl_buffer_mult

            # 方向的中率用の5日後終値
            if len(future_bars) >= HOLD_DAYS:
                exit_close = float(future_bars.iloc[HOLD_DAYS - 1]['Close'])
                fwd_return = (exit_close - entry_price) / entry_price
                hit_dir = (direction == "買い" and fwd_return > 0) or \
                          (direction == "売り" and fwd_return < 0)
                direction_records.append(hit_dir)

            # ── OHLC累積スコア計算（TP/SLに関係なく5日分集計）──
            ohlc_bars = future_bars.head(HOLD_DAYS)
            daily_ohlc = []
            for _, bar in ohlc_bars.iterrows():
                h = float(bar['High'])
                l = float(bar['Low'])
                c = float(bar['Close'])
                if direction == "買い":
                    score = (h - entry_price) + (l - entry_price) + (c - entry_price)
                else:
                    score = (entry_price - h) + (entry_price - l) + (entry_price - c)
                daily_ohlc.append(score / entry_price * 100)
            # 足りない日はNaN埋め
            while len(daily_ohlc) < HOLD_DAYS:
                daily_ohlc.append(float('nan'))
            cumulative_ohlc = sum(v for v in daily_ohlc if not np.isnan(v))

            if tp_level is None:
                continue

            # TP/SL判定（最大5日間の高値・安値でチェック）
            hold_bars = future_bars.head(HOLD_DAYS)
            result    = None  # 'win' / 'loss' / 'forced'
            exit_price = None
            mfe = mae = 0.0

            for _, bar in hold_bars.iterrows():
                bar_high = float(bar['High'])
                bar_low  = float(bar['Low'])

                if direction == "買い":
                    bar_mfe = bar_high - entry_price
                    bar_mae = entry_price - bar_low
                    hit_tp  = bar_high >= tp_level
                    hit_sl  = bar_low  <= sl_level
                else:
                    bar_mfe = entry_price - bar_low
                    bar_mae = bar_high - entry_price
                    hit_tp  = bar_low  <= tp_level
                    hit_sl  = bar_high >= sl_level

                mfe = max(mfe, bar_mfe)
                mae = max(mae, bar_mae)

                if hit_sl or hit_tp:
                    if hit_sl and hit_tp:
                        result = 'loss'       # 同日ヒット → 損切り扱い
                        exit_price = sl_level
                    elif hit_sl:
                        result = 'loss'
                        exit_price = sl_level
                    else:
                        result = 'win'
                        exit_price = tp_level
                    break

            if result is None:
                # 5日以内にTP/SLどちらも未到達 → 5日目引け値で強制決済
                exit_price = float(hold_bars.iloc[-1]['Close'])
                temp_pnl = (exit_price - entry_price) if direction == "買い" else (entry_price - exit_price)
                result = 'forced_win' if temp_pnl > 0 else 'forced_loss'

            if direction == "買い":
                pnl     = exit_price - entry_price
                rr_real = pnl / abs(sl_level - entry_price) if sl_level != entry_price else 0.0
            else:
                pnl     = entry_price - exit_price
                rr_real = pnl / abs(sl_level - entry_price) if sl_level != entry_price else 0.0

            pnl_pct = pnl / entry_price * 100

            trade_records.append({
                'eval_dt':        eval_dt.strftime('%Y-%m-%d'),
                'code':           code,
                'direction':      direction,
                'entry':          round(entry_price, 1),
                'tp':             round(tp_level, 1),
                'sl':             round(sl_level, 1),
                'exit':           round(exit_price, 1),
                'result':         result,
                'pnl_pct':        round(pnl_pct, 2),
                'rr_real':        round(rr_real, 2),
                'mfe_pct':        round(mfe / entry_price * 100, 2),
                'mae_pct':        round(mae / entry_price * 100, 2),
                'd1_ohlc_pct':    round(daily_ohlc[0], 3) if not np.isnan(daily_ohlc[0]) else '',
                'd2_ohlc_pct':    round(daily_ohlc[1], 3) if not np.isnan(daily_ohlc[1]) else '',
                'd3_ohlc_pct':    round(daily_ohlc[2], 3) if not np.isnan(daily_ohlc[2]) else '',
                'd4_ohlc_pct':    round(daily_ohlc[3], 3) if not np.isnan(daily_ohlc[3]) else '',
                'd5_ohlc_pct':    round(daily_ohlc[4], 3) if not np.isnan(daily_ohlc[4]) else '',
                'cumulative_ohlc':  round(cumulative_ohlc, 3),
                'sl_buffer_mult':   sl_buffer_mult,
                'ma_divergence':    round(ma_div, 3),
                'trend_aligned':    trend_aligned,
                'swing_rank':       rank,
            })

    return trade_records, direction_records


def calc_metrics(trade_records, direction_records):
    if not trade_records:
        return {}

    tp_wins  = [t for t in trade_records if t['result'] == 'win']
    sl_losses= [t for t in trade_records if t['result'] == 'loss']
    f_wins   = [t for t in trade_records if t['result'] == 'forced_win']
    f_losses = [t for t in trade_records if t['result'] == 'forced_loss']
    wins     = tp_wins  + f_wins
    losses   = sl_losses + f_losses
    n        = len(trade_records)

    win_rate     = len(wins) / n * 100
    gross_profit = sum(t['pnl_pct'] for t in wins)
    gross_loss   = abs(sum(t['pnl_pct'] for t in losses))
    pf           = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_rr       = np.mean([t['rr_real'] for t in trade_records])
    avg_mfe      = np.mean([t['mfe_pct'] for t in trade_records])
    avg_mae      = np.mean([t['mae_pct'] for t in trade_records])
    avg_pnl      = np.mean([t['pnl_pct'] for t in trade_records])

    dir_rate = sum(direction_records) / len(direction_records) * 100 if direction_records else 0.0

    # ── OHLC累積スコア分析 ──
    def _ohlc_list(records):
        return [t['cumulative_ohlc'] for t in records if isinstance(t.get('cumulative_ohlc'), float)]

    all_ohlc   = _ohlc_list(trade_records)
    sl_ohlc    = _ohlc_list(sl_losses)
    win_ohlc   = _ohlc_list(wins)
    forc_ohlc  = _ohlc_list(f_losses)

    avg_ohlc_all  = round(np.mean(all_ohlc),  3) if all_ohlc  else 0.0
    avg_ohlc_sl   = round(np.mean(sl_ohlc),   3) if sl_ohlc   else 0.0
    avg_ohlc_win  = round(np.mean(win_ohlc),  3) if win_ohlc  else 0.0
    avg_ohlc_forc = round(np.mean(forc_ohlc), 3) if forc_ohlc else 0.0

    # SL到達のうちcumulative_ohlcがプラスだった件数（SLが早すぎた可能性）
    sl_ohlc_positive = sum(1 for v in sl_ohlc if v > 0)
    sl_ohlc_pos_rate = round(sl_ohlc_positive / len(sl_ohlc) * 100, 1) if sl_ohlc else 0.0

    # 日別平均OHLCスコア
    day_avgs = []
    for d in range(1, HOLD_DAYS + 1):
        key = f'd{d}_ohlc_pct'
        vals = [t[key] for t in trade_records if isinstance(t.get(key), float)]
        day_avgs.append(round(np.mean(vals), 3) if vals else 0.0)

    return {
        'n':                  n,
        'wins':               len(wins),
        'tp_wins':            len(tp_wins),
        'f_wins':             len(f_wins),
        'losses':             len(losses),
        'sl_losses':          len(sl_losses),
        'f_losses':           len(f_losses),
        'win_rate':           round(win_rate, 1),
        'pf':                 round(pf, 2),
        'avg_rr':             round(avg_rr, 2),
        'avg_mfe':            round(avg_mfe, 2),
        'avg_mae':            round(avg_mae, 2),
        'avg_pnl':            round(avg_pnl, 2),
        'dir_rate':           round(dir_rate, 1),
        'dir_n':              len(direction_records),
        'avg_ohlc_all':       avg_ohlc_all,
        'avg_ohlc_win':       avg_ohlc_win,
        'avg_ohlc_sl':        avg_ohlc_sl,
        'avg_ohlc_forc_loss': avg_ohlc_forc,
        'sl_ohlc_pos_rate':   sl_ohlc_pos_rate,
        'sl_ohlc_positive':   sl_ohlc_positive,
        'ohlc_day_avgs':      day_avgs,
    }


def _group_metrics(trades):
    """トレードリストから勝率・PF・SL率・平均損益を計算"""
    n = len(trades)
    if n == 0:
        return {'n': 0, 'win_rate': 0.0, 'pf': 0.0, 'sl_rate': 0.0, 'avg_pnl': 0.0}
    wins   = [t for t in trades if t['result'] in ('win', 'forced_win')]
    losses = [t for t in trades if t['result'] in ('loss', 'forced_loss')]
    sl_hits = [t for t in trades if t['result'] == 'loss']
    gp = sum(t['pnl_pct'] for t in wins)
    gl = abs(sum(t['pnl_pct'] for t in losses))
    return {
        'n':        n,
        'win_rate': round(len(wins) / n * 100, 1),
        'pf':       round(gp / gl, 2) if gl > 0 else float('inf'),
        'sl_rate':  round(len(sl_hits) / n * 100, 1),
        'avg_pnl':  round(sum(t['pnl_pct'] for t in trades) / n, 2),
    }


def calc_direction_trend_analysis(trade_records):
    """
    方向（買い/売り）× トレンド一致/不一致 の4グループ別成績
    """
    groups = {
        '買い×順張り': [],
        '買い×逆張り': [],
        '売り×順張り': [],
        '売り×逆張り': [],
    }
    for t in trade_records:
        d  = t['direction']
        ta = t.get('trend_aligned', True)
        if d == "買い":
            groups['買い×順張り' if ta else '買い×逆張り'].append(t)
        else:
            groups['売り×順張り' if ta else '売り×逆張り'].append(t)

    return {k: _group_metrics(v) for k, v in groups.items()}


def run_hensho_analysis(all_data_dict, eval_dates, nikkei_data, threshold=5):
    """
    変調検知バックテスト:
    - ハイブリッドMAフィルター通過件数を日別に集計
    - threshold未満の日を「変調日」として特定
    - 変調日の敗者復活候補（逆方向×MA5/25のみ一致）を収集・成績計算
    - 日経終値と5日後リターンを記録（転換点との重ね合わせ用）
    """
    import copy
    daily_records   = []   # 日別サマリー
    repechage_trades = []  # 敗者復活候補の個別トレード

    for eval_dt in eval_dates:
        eval_ts = pd.Timestamp(eval_dt).normalize()

        # ── データスライス＆売買代金Top50 ──
        slices = {}
        for code, df_full in all_data_dict.items():
            df_slice = df_full[df_full.index.normalize() < eval_ts]
            if len(df_slice) >= HIST_BARS:
                slices[code] = df_slice

        vol_scores = []
        for code, df_s in slices.items():
            recent5 = df_s.tail(5)
            trade_val = (recent5['Volume'] * recent5['Close']).mean() / 1e8
            vol_scores.append((code, trade_val))
        vol_scores.sort(key=lambda x: x[1], reverse=True)
        top_codes = {code for code, _ in vol_scores[:TOP_VOL]}

        raws = []
        for code in top_codes:
            raw = calc_indicators(slices[code])
            if raw is None:
                continue
            raw['code'] = code
            raws.append(raw)

        if len(raws) < 3:
            continue

        raws_copy = copy.deepcopy(raws)
        score_swing_new(raws_copy)
        candidates = apply_exclusions(raws_copy)

        # ── 日経MA判定 ──
        trend_short = calc_nikkei_trend(nikkei_data, eval_ts, fast=5,  slow=25)
        trend_long  = calc_nikkei_trend(nikkei_data, eval_ts, fast=25, slow=75)
        if trend_short is None or trend_long is None:
            continue

        nikkei_dir = trend_short if trend_short == trend_long else None  # None=転換期

        # ── ハイブリッドMAフィルター ──
        filtered_normal = []
        for r in candidates:
            direction  = r.get('swing_direction')
            ma25_75_ok = (r.get('ma_divergence', 0) > 0) if direction == '買い' \
                         else (r.get('ma_divergence', 0) < 0)
            ma5_25_ok  = r.get('ma5_25_bull', False) if direction == '買い' \
                         else (not r.get('ma5_25_bull', True))
            stock_ok   = ma25_75_ok and ma5_25_ok
            if nikkei_dir is None:
                if stock_ok:
                    filtered_normal.append(r)
            elif direction == nikkei_dir:
                filtered_normal.append(r)
            elif stock_ok:
                filtered_normal.append(r)

        normal_count = len(filtered_normal)

        # ── 日経終値・5日後リターン ──
        nk_past = nikkei_data[nikkei_data.index.normalize() <= eval_ts]
        nk_future = nikkei_data[nikkei_data.index.normalize() > eval_ts]
        nk_close = float(nk_past['Close'].iloc[-1]) if len(nk_past) > 0 else None
        nk_5d_return = None
        if nk_close and len(nk_future) >= 5:
            nk_5d_return = round((float(nk_future['Close'].iloc[4]) - nk_close) / nk_close * 100, 2)

        # ── 変調日：敗者復活候補を収集 ──
        is_hensho = (normal_count < threshold) and (nikkei_dir is not None)
        repechage_count = 0
        repechage_pnl_list = []

        if is_hensho:
            counter_dir = '買い' if nikkei_dir == '売り' else '売り'
            repechage_candidates = sorted(
                [r for r in candidates if r.get('swing_direction') == counter_dir
                 and (r.get('ma5_25_bull', False) if counter_dir == '買い'
                      else not r.get('ma5_25_bull', True))],
                key=lambda x: x['swing_score'], reverse=True
            )[:10 - normal_count]  # 不足分だけ

            for r in repechage_candidates:
                code      = r['code']
                direction = r.get('swing_direction')
                tp_level  = r.get('swing_tp')
                if tp_level is None:
                    continue
                df_full   = all_data_dict.get(code)
                if df_full is None:
                    continue
                future_bars = df_full[df_full.index.normalize() > eval_ts]
                if len(future_bars) < 1:
                    continue

                entry_price = float(future_bars.iloc[0]['Open'])
                tp_dist = abs(tp_level - entry_price)
                sl_level = (entry_price - tp_dist) if direction == '買い' \
                           else (entry_price + tp_dist)  # RR=1.0

                # TP/SL判定（5日間）
                hold_bars  = future_bars.head(HOLD_DAYS)
                result     = None
                exit_price = None
                for _, bar in hold_bars.iterrows():
                    bh, bl = float(bar['High']), float(bar['Low'])
                    hit_tp = (bh >= tp_level) if direction == '買い' else (bl <= tp_level)
                    hit_sl = (bl <= sl_level) if direction == '買い' else (bh >= sl_level)
                    if hit_sl or hit_tp:
                        if hit_sl and hit_tp:
                            result, exit_price = 'loss', sl_level
                        elif hit_sl:
                            result, exit_price = 'loss', sl_level
                        else:
                            result, exit_price = 'win', tp_level
                        break
                if result is None:
                    exit_price = float(hold_bars.iloc[-1]['Close'])
                    temp = (exit_price - entry_price) if direction == '買い' \
                           else (entry_price - exit_price)
                    result = 'forced_win' if temp > 0 else 'forced_loss'

                pnl = (exit_price - entry_price) if direction == '買い' \
                      else (entry_price - exit_price)
                pnl_pct = pnl / entry_price * 100

                repechage_count += 1
                repechage_pnl_list.append(pnl_pct)
                repechage_trades.append({
                    'date':         eval_dt.strftime('%Y-%m-%d'),
                    'code':         code,
                    'direction':    direction,
                    'nikkei_dir':   nikkei_dir,
                    'normal_count': normal_count,
                    'entry':        round(entry_price, 1),
                    'tp':           round(tp_level, 1),
                    'sl':           round(sl_level, 1),
                    'exit':         round(exit_price, 1),
                    'result':       result,
                    'pnl_pct':      round(pnl_pct, 2),
                    'ma5_25_bull':  r.get('ma5_25_bull'),
                    'ma_divergence': round(r.get('ma_divergence', 0), 3),
                    'swing_score':  r.get('swing_score', 0),
                })

        gains  = sum(p for p in repechage_pnl_list if p > 0)
        losses = abs(sum(p for p in repechage_pnl_list if p < 0))
        rep_pf = round(gains / losses, 2) if losses > 0 else (float('inf') if gains > 0 else None)

        daily_records.append({
            'date':             eval_dt.strftime('%Y-%m-%d'),
            'nikkei_close':     round(nk_close, 0) if nk_close else None,
            'nikkei_5d_return': nk_5d_return,
            'nikkei_dir':       nikkei_dir or '転換期',
            'normal_count':     normal_count,
            'is_hensho':        is_hensho,
            'repechage_count':  repechage_count,
            'repechage_pf':     rep_pf,
            'repechage_avg_pnl': round(sum(repechage_pnl_list)/len(repechage_pnl_list), 2)
                                  if repechage_pnl_list else None,
        })

    return daily_records, repechage_trades


def calc_nikkei_trend(df_nikkei, eval_ts, fast=25, slow=75):
    """
    評価日時点の日経平均トレンドを返す
    fast MA > slow MA → '買い'（上昇トレンド）
    fast MA < slow MA → '売り'（下降トレンド）
    """
    df = df_nikkei[df_nikkei.index.normalize() < eval_ts]
    if len(df) < slow:
        return None  # データ不足 → フィルターなし
    closes = df['Close'].values.astype(float)
    ma_fast = closes[-fast:].mean()
    ma_slow = closes[-slow:].mean()
    return '買い' if ma_fast >= ma_slow else '売り'


def calc_monthly_analysis(trade_records):
    """
    月別 × 方向（買い/売り）の成績分析
    """
    from collections import defaultdict
    months = sorted(set(t['eval_dt'][:7] for t in trade_records))
    result = {}
    for ym in months:
        month_trades = [t for t in trade_records if t['eval_dt'][:7] == ym]
        buy_trades   = [t for t in month_trades if t['direction'] == '買い']
        sell_trades  = [t for t in month_trades if t['direction'] == '売り']
        result[ym] = {
            'all':  _group_metrics(month_trades),
            'buy':  _group_metrics(buy_trades),
            'sell': _group_metrics(sell_trades),
        }
    return result


# ─────────────────────────────────────────────
# 指値バックテスト本体
# ─────────────────────────────────────────────

def run_backtest_limit(all_data_dict, eval_dates, rr_target,
                       atr_filter_mult=ATR_FILTER_MULT,
                       entry_window=ENTRY_WINDOW,
                       nikkei_data=None, use_market_filter=False,
                       nikkei_fast=25, nikkei_slow=75):
    """
    指値エントリー方式のバックテスト。

    シグナル日の翌1〜entry_window日の間に
    買い: その日のLOW <= entry_calc
    売り: その日のHIGH >= entry_calc
    で約定。未到達は評価対象外。
    """
    import copy

    trade_records  = []
    stats = {'unfilled': 0, 'atr_filtered': 0, 'dir_oor': 0, 'total_signals': 0}

    for eval_dt in eval_dates:
        eval_ts = pd.Timestamp(eval_dt).normalize()

        slices = {}
        for code, df_full in all_data_dict.items():
            df_slice = df_full[df_full.index.normalize() < eval_ts]
            if len(df_slice) >= HIST_BARS:
                slices[code] = df_slice

        vol_scores = [(code, (df_s.tail(5)['Volume'] * df_s.tail(5)['Close']).mean() / 1e8)
                      for code, df_s in slices.items()]
        vol_scores.sort(key=lambda x: x[1], reverse=True)
        top_codes = {code for code, _ in vol_scores[:TOP_VOL]}

        raws = []
        for code in top_codes:
            raw = calc_indicators(slices[code])
            if raw is None:
                continue
            raw['code'] = code
            raws.append(raw)

        if len(raws) < 3:
            continue

        raws_copy = copy.deepcopy(raws)
        score_swing_new(raws_copy)

        candidates = [r for r in raws_copy if r.get('swing_rr_valid', False)]
        candidates = apply_exclusions(candidates)
        if not candidates:
            continue

        if use_market_filter and nikkei_data is not None:
            nikkei_trend = calc_nikkei_trend(nikkei_data, eval_ts,
                                             fast=nikkei_fast, slow=nikkei_slow)
            if nikkei_trend is not None:
                candidates = [r for r in candidates
                              if r['swing_direction'] == nikkei_trend]
            if not candidates:
                continue

        top = sorted(candidates, key=lambda x: x['swing_score'], reverse=True)[:TOP_N]
        stats['total_signals'] += len(top)

        for rank, s in enumerate(top, 1):
            code      = s['code']
            direction = s['swing_direction']
            curr      = s['curr']          # シグナル日終値
            atr       = s['atr_val']
            sl_zone   = s['swing_sl']      # サポレジゾーン境界
            tp_zone   = s['swing_tp']

            if sl_zone is None or tp_zone is None:
                continue

            sl_actual, tp_actual, entry_calc = calc_entry_params(
                sl_zone, tp_zone, rr_target, direction)

            # エントリーが現在値の正しい方向にあるか確認
            if direction == '買い' and entry_calc >= curr:
                stats['dir_oor'] += 1
                continue
            if direction == '売り' and entry_calc <= curr:
                stats['dir_oor'] += 1
                continue

            # ATRフィルター：エントリー価格がcurrから遠すぎる場合は除外
            if atr > 0 and abs(entry_calc - curr) > atr_filter_mult * atr:
                stats['atr_filtered'] += 1
                continue

            df_full     = all_data_dict[code]
            future_bars = df_full[df_full.index.normalize() > eval_ts]
            if len(future_bars) < 1:
                continue

            # ── エントリー到達チェック（最大 entry_window 日）──
            entry_triggered  = False
            entry_day_offset = None
            for day_idx, (_, bar) in enumerate(future_bars.head(entry_window).iterrows()):
                if direction == '買い' and float(bar['Low']) <= entry_calc:
                    entry_triggered  = True
                    entry_day_offset = day_idx
                    break
                elif direction == '売り' and float(bar['High']) >= entry_calc:
                    entry_triggered  = True
                    entry_day_offset = day_idx
                    break

            if not entry_triggered:
                stats['unfilled'] += 1
                continue

            entry_price = entry_calc

            # ── TP/SL判定（エントリー日から最大HOLD_DAYS日）──
            hold_bars = future_bars.iloc[entry_day_offset: entry_day_offset + HOLD_DAYS]
            if len(hold_bars) < 1:
                continue

            result     = None
            exit_price = None
            mfe = mae  = 0.0
            daily_data = []   # [(日次最大含み益%, 日次終値リターン%)]

            for _, bar in hold_bars.iterrows():
                bar_high  = float(bar['High'])
                bar_low   = float(bar['Low'])
                bar_close = float(bar['Close'])

                if direction == '買い':
                    bar_mfe = bar_high - entry_price
                    bar_mae = entry_price - bar_low
                    hit_tp  = bar_high >= tp_actual
                    hit_sl  = bar_low  <= sl_actual
                    d_mfe   = (bar_high  - entry_price) / entry_price * 100
                    d_ret   = (bar_close - entry_price) / entry_price * 100
                else:
                    bar_mfe = entry_price - bar_low
                    bar_mae = bar_high - entry_price
                    hit_tp  = bar_low  <= tp_actual
                    hit_sl  = bar_high >= sl_actual
                    d_mfe   = (entry_price - bar_low)   / entry_price * 100
                    d_ret   = (entry_price - bar_close) / entry_price * 100

                mfe = max(mfe, bar_mfe)
                mae = max(mae, bar_mae)
                daily_data.append((round(d_mfe, 3), round(d_ret, 3)))

                if hit_sl or hit_tp:
                    if hit_sl and hit_tp:
                        result = 'loss';      exit_price = sl_actual
                    elif hit_sl:
                        result = 'loss';      exit_price = sl_actual
                    else:
                        result = 'win';       exit_price = tp_actual
                    break

            if result is None:
                exit_price = float(hold_bars.iloc[-1]['Close'])
                pnl_tmp    = (exit_price - entry_price) if direction == '買い' \
                             else (entry_price - exit_price)
                result     = 'forced_win' if pnl_tmp > 0 else 'forced_loss'

            if direction == '買い':
                pnl     = exit_price - entry_price
            else:
                pnl     = entry_price - exit_price
            pnl_pct = pnl / entry_price * 100
            rr_real = pnl / abs(sl_actual - entry_price) \
                      if sl_actual != entry_price else 0.0

            # 日次データを列に展開（不足分はnan埋め）
            while len(daily_data) < HOLD_DAYS:
                daily_data.append((float('nan'), float('nan')))

            trade_records.append({
                'rr_target':  rr_target,
                'eval_dt':    eval_dt.strftime('%Y-%m-%d'),
                'code':       code,
                'direction':  direction,
                'curr':       round(curr, 1),
                'entry':      round(entry_price, 1),
                'tp':         round(tp_actual, 1),
                'sl':         round(sl_actual, 1),
                'exit':       round(exit_price, 1),
                'result':     result,
                'pnl_pct':    round(pnl_pct, 2),
                'rr_real':    round(rr_real, 2),
                'mfe_pct':    round(mfe / entry_price * 100, 2),
                'mae_pct':    round(mae / entry_price * 100, 2),
                'entry_day':  entry_day_offset + 1,
                'swing_rank': rank,
                **{f'd{i+1}_mfe': daily_data[i][0] for i in range(HOLD_DAYS)},
                **{f'd{i+1}_ret': daily_data[i][1] for i in range(HOLD_DAYS)},
            })

    return trade_records, stats


def calc_metrics_limit(trade_records, stats):
    """指値バックテスト用メトリクス計算"""
    if not trade_records:
        return {}

    n        = len(trade_records)
    wins     = [t for t in trade_records if t['result'] in ('win', 'forced_win')]
    losses   = [t for t in trade_records if t['result'] in ('loss', 'forced_loss')]
    tp_wins  = [t for t in trade_records if t['result'] == 'win']
    sl_losses= [t for t in trade_records if t['result'] == 'loss']

    win_rate     = len(wins) / n * 100
    gross_profit = sum(t['pnl_pct'] for t in wins)
    gross_loss   = abs(sum(t['pnl_pct'] for t in losses))
    pf           = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_pnl      = np.mean([t['pnl_pct'] for t in trade_records])
    avg_mfe      = np.mean([t['mfe_pct'] for t in trade_records])
    avg_mae      = np.mean([t['mae_pct'] for t in trade_records])

    total_signals = stats['total_signals']
    entry_rate    = n / total_signals * 100 if total_signals > 0 else 0.0

    # 日次含み益（MFE）・終値リターン平均
    day_mfe_avgs = []
    day_ret_avgs = []
    for d in range(1, HOLD_DAYS + 1):
        mfe_vals = [t[f'd{d}_mfe'] for t in trade_records
                    if isinstance(t.get(f'd{d}_mfe'), float)
                    and not np.isnan(t[f'd{d}_mfe'])]
        ret_vals = [t[f'd{d}_ret'] for t in trade_records
                    if isinstance(t.get(f'd{d}_ret'), float)
                    and not np.isnan(t[f'd{d}_ret'])]
        day_mfe_avgs.append(round(np.mean(mfe_vals), 3) if mfe_vals else 0.0)
        day_ret_avgs.append(round(np.mean(ret_vals), 3) if ret_vals else 0.0)

    return {
        'n':             n,
        'wins':          len(wins),
        'tp_wins':       len(tp_wins),
        'sl_losses':     len(sl_losses),
        'win_rate':      round(win_rate, 1),
        'pf':            round(pf, 2),
        'avg_pnl':       round(avg_pnl, 2),
        'avg_mfe':       round(avg_mfe, 2),
        'avg_mae':       round(avg_mae, 2),
        'sl_rate':       round(len(sl_losses) / n * 100, 1),
        'tp_rate':       round(len(tp_wins)   / n * 100, 1),
        'entry_rate':    round(entry_rate, 1),
        'total_signals': total_signals,
        'unfilled':      stats['unfilled'],
        'atr_filtered':  stats['atr_filtered'],
        'day_mfe_avgs':  day_mfe_avgs,
        'day_ret_avgs':  day_ret_avgs,
    }


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("スイング推奨バックテスト（本格版）")
    print("エントリー: 翌営業日寄り付き")
    print("決済: TP(HVNベース) / SL(ATRベース) 先着、上限5日")
    print("=" * 60)

    # ──── データ取得 ────
    end_dt   = datetime.now() - timedelta(days=5)
    start_dt = end_dt - timedelta(days=400)

    print(f"\n[1/3] 株価データ取得中（{len(UNIVERSE)}銘柄）...")
    all_data = {}
    for code in UNIVERSE:
        try:
            df = yf.Ticker(f"{code}.T").history(
                start=start_dt.strftime('%Y-%m-%d'),
                end=(end_dt + timedelta(days=30)).strftime('%Y-%m-%d'),
                auto_adjust=False
            )
            if len(df) >= HIST_BARS + HOLD_DAYS + 20:
                df.index = df.index.tz_localize(None)
                all_data[code] = df
                print(f"  {code}: {len(df)}本", end="\r")
        except Exception as e:
            print(f"  {code}: 取得失敗 ({e})")
        time.sleep(0.15)
    print(f"\n取得成功: {len(all_data)}銘柄")

    # ──── 日経平均データ取得 ────
    print("  日経平均（^N225）取得中...")
    nikkei_data = None
    try:
        df_n = yf.Ticker("^N225").history(
            start=start_dt.strftime('%Y-%m-%d'),
            end=(end_dt + timedelta(days=30)).strftime('%Y-%m-%d'),
            auto_adjust=False
        )
        if len(df_n) >= 75:
            df_n.index = df_n.index.tz_localize(None)
            nikkei_data = df_n
            print(f"  ^N225: {len(df_n)}本取得")
        else:
            print("  ^N225: データ不足、市場フィルターなしで実行")
    except Exception as e:
        print(f"  ^N225: 取得失敗 ({e})、市場フィルターなしで実行")

    # ──── 評価日生成（過去4ヶ月・5営業日おき）────
    print("\n[2/3] 評価日生成中...")
    eval_end   = end_dt - timedelta(days=HOLD_DAYS + 5)
    eval_start = eval_end - timedelta(days=120)

    cal_df = next(iter(all_data.values()))
    trading_days = list(cal_df[(cal_df.index >= pd.Timestamp(eval_start)) &
                               (cal_df.index <= pd.Timestamp(eval_end))].index)
    eval_dates = [d.to_pydatetime() for d in trading_days[::1]]

    if not eval_dates:
        print("ERROR: 評価日が生成できませんでした")
        exit(1)

    print(f"  評価期間: {eval_dates[0].strftime('%Y-%m-%d')} ～ {eval_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  評価ポイント数: {len(eval_dates)}")

    # ──── バックテスト実行：SLバッファ×市場フィルター ────
    print(f"\n[3/3] バックテスト実行中...")
    all_metrics = {}
    all_trades  = {}
    base_direction_records = None

    # SLバッファ比較（フィルターなし）
    for mult in SL_BUFFER_MULTS:
        print(f"  [フィルターなし] SL×{mult:.1f} 実行中...", end="\r")
        trades, dir_recs = run_backtest(all_data, eval_dates, sl_buffer_mult=mult)
        all_metrics[mult] = calc_metrics(trades, dir_recs)
        all_trades[mult]  = trades
        if base_direction_records is None:
            base_direction_records = dir_recs

    # 市場フィルターあり：MAパターン別比較
    MA_PATTERNS = [
        (5,  25, "MA5/25"),
        (10, 25, "MA10/25"),
        (25, 75, "MA25/75"),
    ]
    filter_results = {}
    for fast, slow, label in MA_PATTERNS:
        print(f"  [市場フィルター {label}] 実行中...", end="\r")
        trades_f, dir_recs_f = run_backtest(
            all_data, eval_dates, sl_buffer_mult=1.0,
            nikkei_data=nikkei_data, use_market_filter=True,
            nikkei_fast=fast, nikkei_slow=slow
        )
        filter_results[label] = {
            'trades':  trades_f,
            'metrics': calc_metrics(trades_f, dir_recs_f),
            'monthly': calc_monthly_analysis(trades_f),
        }

    # 日経デュアルMAフィルター（MA5/25 AND MA25/75 日経で両方一致のみ）
    print(f"  [日経デュアルMA] 実行中...", end="\r")
    trades_dual, dir_recs_dual = run_backtest(
        all_data, eval_dates, sl_buffer_mult=1.0,
        nikkei_data=nikkei_data, use_market_filter=True,
        dual_ma_filter=True
    )
    filter_results["日経デュアルMA"] = {
        'trades':  trades_dual,
        'metrics': calc_metrics(trades_dual, dir_recs_dual),
        'monthly': calc_monthly_analysis(trades_dual),
    }

    # 個別銘柄デュアルMAフィルター（各銘柄のMA5/25 AND MA25/75 が両方一致のみ通過）
    print(f"  [銘柄デュアルMA] 実行中...", end="\r")
    trades_sdual, dir_recs_sdual = run_backtest(
        all_data, eval_dates, sl_buffer_mult=1.0,
        stock_dual_ma=True
    )
    filter_results["銘柄デュアルMA"] = {
        'trades':  trades_sdual,
        'metrics': calc_metrics(trades_sdual, dir_recs_sdual),
        'monthly': calc_monthly_analysis(trades_sdual),
    }

    # ハイブリッドMAフィルター（日経一致→日経方向フィルター、転換期→銘柄デュアルMAフォールバック）
    print(f"  [ハイブリッドMA] 実行中...", end="\r")
    trades_hybrid, dir_recs_hybrid = run_backtest(
        all_data, eval_dates, sl_buffer_mult=1.0,
        nikkei_data=nikkei_data, hybrid_ma=True
    )
    filter_results["ハイブリッドMA"] = {
        'trades':  trades_hybrid,
        'metrics': calc_metrics(trades_hybrid, dir_recs_hybrid),
        'monthly': calc_monthly_analysis(trades_hybrid),
    }
    # 後の表示用に最初のパターンを代表として保持
    trades_f  = filter_results['MA5/25']['trades']
    metrics_f = filter_results['MA5/25']['metrics']
    print()

    # ──── 比較表表示 ────
    print("\n" + "=" * 72)
    print("【SLバッファ比較】")
    print("=" * 72)
    header = f"{'指標':<22}" + "".join(f"{'×'+str(m):>8}" for m in SL_BUFFER_MULTS)
    print(header)
    print("-" * 72)

    def row(label, key, fmt=".1f"):
        vals = "".join(f"{all_metrics[m][key]:{'>8'+fmt}}" for m in SL_BUFFER_MULTS)
        print(f"{label:<22}{vals}")

    row("勝率(%)",           "win_rate")
    row("プロフィットF",     "pf",       ".2f")
    row("SL到達件数",        "sl_losses", "d")
    row("平均損益(%)",       "avg_pnl",  ".2f")
    row("平均MFE(%)",        "avg_mfe",  ".2f")
    row("平均MAE(%)",        "avg_mae",  ".2f")
    row("SL早すぎ疑い率(%)", "sl_ohlc_pos_rate")
    print("-" * 72)

    # ──── ベースライン(×1.0)の詳細表示 ────
    base = all_metrics[1.0]
    print(f"\n【詳細：SL×1.0（現状）】")
    print(f"  総トレード数  : {base['n']}")
    print(f"  勝ち計        : {base['wins']}  (TP到達:{base['tp_wins']} + 強制益:{base['f_wins']})")
    print(f"  負け計        : {base['losses']}  (SL到達:{base['sl_losses']} + 強制損:{base['f_losses']})")
    print(f"  方向的中率(5日後終値): {base['dir_rate']:.1f}%")
    print()
    print("  【OHLC分析】")
    print(f"  全トレード平均  : {base['avg_ohlc_all']:+.3f}%")
    print(f"  SL到達平均      : {base['avg_ohlc_sl']:+.3f}%")
    print(f"  SL到達でOHLCプラス: {base['sl_ohlc_positive']}件/{base['sl_losses']}件 ({base['sl_ohlc_pos_rate']:.1f}%)")
    day_labels = ["1日目", "2日目", "3日目", "4日目", "5日目"]
    for label, val in zip(day_labels, base['ohlc_day_avgs']):
        print(f"    {label}: {val:+.3f}%")

    # ──── 市場フィルター MAパターン比較表示 ────
    base = all_metrics[1.0]
    ALL_FILTER_LABELS = [lb for _, _, lb in MA_PATTERNS] + ["日経デュアルMA", "銘柄デュアルMA", "ハイブリッドMA"]
    sep_w = 20 + 12 * (1 + len(ALL_FILTER_LABELS))
    print("\n" + "=" * sep_w)
    print("【市場環境フィルター MAパターン比較（SL×1.0）】")
    print("=" * sep_w)
    col_w = 12
    header = f"{'指標':<20}{'なし':>{col_w}}" + "".join(f"{lb:>{col_w}}" for lb in ALL_FILTER_LABELS)
    print(header)
    print("-" * sep_w)

    def frow(label, key, fmt=".1f"):
        base_v = f"{base[key]:{fmt}}"
        vals = "".join(f"{f'{filter_results[lb]['metrics'][key]:{fmt}}':>{col_w}}"
                       for lb in ALL_FILTER_LABELS)
        print(f"{label:<20}{base_v:>{col_w}}{vals}")

    def frow_calc(label, fn):
        base_v = fn(base)
        vals = "".join(f"{fn(filter_results[lb]['metrics']):>{col_w}}"
                       for lb in ALL_FILTER_LABELS)
        print(f"{label:<20}{base_v:>{col_w}}{vals}")

    frow("総トレード数",   "n",        "d")
    frow("勝率(%)",       "win_rate",  ".1f")
    frow("プロフィットF", "pf",        ".2f")
    frow("SL到達件数",    "sl_losses", "d")
    frow_calc("SL到達率(%)",
              lambda m: f"{m['sl_losses']/m['n']*100:.1f}" if m['n'] > 0 else "-")
    frow("平均損益(%)",   "avg_pnl",   "+.2f")
    frow("方向的中率(%)", "dir_rate",  ".1f")
    print("-" * sep_w)
    print("  ※日経デュアルMA = 日経MA5/25 AND MA25/75 両方一致の日のみトレード（不一致＝転換期は休む）")
    print("  ※銘柄デュアルMA = 各銘柄のMA5/25 AND MA25/75 両方一致の銘柄のみトレード")
    print("  ※ハイブリッドMA = 日経一致時→日経方向フィルター、転換期→銘柄デュアルMAフォールバック")

    # 各パターンの月別買い/売り件数を表示
    # MA25/75・日経デュアルMA・銘柄デュアルMA・ハイブリッドMAの月別詳細表示
    for lb in ["MA25/75", "日経デュアルMA", "銘柄デュアルMA", "ハイブリッドMA"]:
        monthly_f = filter_results[lb]['monthly']
        print(f"\n【月別成績（{lb}フィルター）】")
        print(f"  {'月':<8}{'全件':>5}{'全PF':>6}  {'買い件':>6}{'買PF':>6}  {'売り件':>6}{'売PF':>6}")
        print("  " + "-" * 52)
        for ym, g in monthly_f.items():
            a, b, s = g['all'], g['buy'], g['sell']
            def pf_s(v): return f"{v:.2f}" if v != float('inf') else " inf"
            print(f"  {ym:<8}{a['n']:>5}{pf_s(a['pf']):>6}  "
                  f"{b['n']:>6}{pf_s(b['pf']):>6}  "
                  f"{s['n']:>6}{pf_s(s['pf']):>6}")
        print("  " + "-" * 52)

    # ──── 方向×トレンド一致/不一致 分析（ベースライン×1.0のみ） ────
    dt_analysis = calc_direction_trend_analysis(all_trades[1.0])

    print("\n" + "=" * 72)
    print("【方向 × トレンド一致/不一致 分析（SL×1.0）】")
    print("  ※順張り = シグナル方向とMA乖離方向が一致")
    print("  ※逆張り = 不一致（ダイバージェンス相当）")
    print("=" * 72)
    hdr = f"{'グループ':<14}{'件数':>6}{'勝率':>8}{'PF':>7}{'SL率':>8}{'平均損益':>9}"
    print(hdr)
    print("-" * 72)
    for label in ['買い×順張り', '買い×逆張り', '売り×順張り', '売り×逆張り']:
        g = dt_analysis[label]
        pf_str = f"{g['pf']:.2f}" if g['pf'] != float('inf') else " inf"
        print(f"{label:<14}{g['n']:>6}{g['win_rate']:>7.1f}%{pf_str:>7}{g['sl_rate']:>7.1f}%{g['avg_pnl']:>+8.2f}%")
    print("-" * 72)

    # ──── 月別分析（ベースライン×1.0のみ） ────
    monthly = calc_monthly_analysis(all_trades[1.0])

    print("\n" + "=" * 80)
    print("【月別成績（SL×1.0）】")
    print("=" * 80)
    print(f"{'月':<8}{'全件':>5}{'全勝率':>7}{'全PF':>6}  "
          f"{'買い件':>6}{'買勝率':>7}{'買PF':>6}  "
          f"{'売り件':>6}{'売勝率':>7}{'売PF':>6}")
    print("-" * 80)
    for ym, g in monthly.items():
        a, b, s = g['all'], g['buy'], g['sell']
        def pf_s(v): return f"{v:.2f}" if v != float('inf') else " inf"
        print(f"{ym:<8}{a['n']:>5}{a['win_rate']:>6.1f}%{pf_s(a['pf']):>6}  "
              f"{b['n']:>6}{b['win_rate']:>6.1f}%{pf_s(b['pf']):>6}  "
              f"{s['n']:>6}{s['win_rate']:>6.1f}%{pf_s(s['pf']):>6}")
    print("-" * 80)

    # ──── MIN_RR比較（ハイブリッドMAベース） ────
    MIN_RR_VALUES = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5]
    rr_results = {}
    print(f"\n  [MIN_RR比較] 実行中...")
    for rr_thresh in MIN_RR_VALUES:
        t, d = run_backtest(
            all_data, eval_dates, sl_buffer_mult=1.0,
            nikkei_data=nikkei_data, hybrid_ma=True,
            min_rr=rr_thresh
        )
        rr_results[rr_thresh] = {
            'trades':  t,
            'metrics': calc_metrics(t, d),
            'monthly': calc_monthly_analysis(t),
        }

    rr_col_w = 10
    rr_labels = [f"RR>={v}" for v in MIN_RR_VALUES]
    rr_labels[0] = "制限なし"
    rr_sep = 20 + rr_col_w * len(MIN_RR_VALUES)
    print("\n" + "=" * rr_sep)
    print("【MIN_RR閾値比較（ハイブリッドMAベース、SL×1.0）】")
    print("=" * rr_sep)
    print(f"{'指標':<20}" + "".join(f"{lb:>{rr_col_w}}" for lb in rr_labels))
    print("-" * rr_sep)

    def rr_row(label, key, fmt=".1f"):
        vals = "".join(f"{f'{rr_results[v]['metrics'][key]:{fmt}}':>{rr_col_w}}" for v in MIN_RR_VALUES)
        print(f"{label:<20}{vals}")

    def rr_row_calc(label, fn):
        vals = "".join(f"{fn(rr_results[v]['metrics']):>{rr_col_w}}" for v in MIN_RR_VALUES)
        print(f"{label:<20}{vals}")

    rr_row("総トレード数",   "n",        "d")
    rr_row("勝率(%)",       "win_rate",  ".1f")
    rr_row("プロフィットF", "pf",        ".2f")
    rr_row_calc("SL到達率(%)",
                lambda m: f"{m['sl_losses']/m['n']*100:.1f}" if m['n'] > 0 else "-")
    rr_row("平均損益(%)",   "avg_pnl",   "+.2f")
    print("-" * rr_sep)
    print("  ※ハイブリッドMA（日経一致→日経方向、転換期→銘柄デュアルMA）ベース")

    # 月別詳細（制限なし vs RR≥1.0）
    for rr_thresh, label in [(0.0, "制限なし"), (1.0, "RR>=1.0")]:
        monthly_r = rr_results[rr_thresh]['monthly']
        print(f"\n【月別成績（{label}）】")
        print(f"  {'月':<8}{'全件':>5}{'全PF':>6}  {'買い件':>6}{'買PF':>6}  {'売り件':>6}{'売PF':>6}")
        print("  " + "-" * 52)
        for ym, g in monthly_r.items():
            a, b, s = g['all'], g['buy'], g['sell']
            def pf_s(v): return f"{v:.2f}" if v != float('inf') else " inf"
            print(f"  {ym:<8}{a['n']:>5}{pf_s(a['pf']):>6}  "
                  f"{b['n']:>6}{pf_s(b['pf']):>6}  "
                  f"{s['n']:>6}{pf_s(s['pf']):>6}")
        print("  " + "-" * 52)

    # ──── RR逆算SL比較（ハイブリッドMAベース） ────
    RR_SL_TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0]
    rr_sl_results = {}
    print(f"\n  [RR逆算SL比較] 実行中...")
    for rr_t in RR_SL_TARGETS:
        t, d = run_backtest(
            all_data, eval_dates, sl_buffer_mult=1.0,
            nikkei_data=nikkei_data, hybrid_ma=True,
            rr_target_sl=rr_t
        )
        rr_sl_results[rr_t] = {
            'trades':  t,
            'metrics': calc_metrics(t, d),
            'monthly': calc_monthly_analysis(t),
        }

    rr_sl_col = 10
    rr_sl_labels = [f"RR={v}" for v in RR_SL_TARGETS]
    rr_sl_sep = 20 + rr_sl_col * len(RR_SL_TARGETS)
    print("\n" + "=" * rr_sl_sep)
    print("【RR逆算SL比較（ハイブリッドMAベース）】")
    print("  ※SL = エントリー ± (TP距離 ÷ 目標RR)")
    print("=" * rr_sl_sep)
    print(f"{'指標':<20}" + "".join(f"{lb:>{rr_sl_col}}" for lb in rr_sl_labels))
    print("-" * rr_sl_sep)

    def rr_sl_row(label, key, fmt=".1f"):
        vals = "".join(f"{f'{rr_sl_results[v]['metrics'][key]:{fmt}}':>{rr_sl_col}}" for v in RR_SL_TARGETS)
        print(f"{label:<20}{vals}")

    def rr_sl_row_calc(label, fn):
        vals = "".join(f"{fn(rr_sl_results[v]['metrics']):>{rr_sl_col}}" for v in RR_SL_TARGETS)
        print(f"{label:<20}{vals}")

    rr_sl_row("総トレード数",   "n",        "d")
    rr_sl_row("勝率(%)",       "win_rate",  ".1f")
    rr_sl_row("プロフィットF", "pf",        ".2f")
    rr_sl_row_calc("SL到達率(%)",
                   lambda m: f"{m['sl_losses']/m['n']*100:.1f}" if m['n'] > 0 else "-")
    rr_sl_row("SL到達件数",    "sl_losses", "d")
    rr_sl_row("平均損益(%)",   "avg_pnl",   "+.2f")
    print("-" * rr_sl_sep)
    print("  ※ハイブリッドMAフィルター適用")

    # 月別詳細（RR=2.0）
    for rr_t, label in [(2.0, "RR=2.0")]:
        monthly_r = rr_sl_results[rr_t]['monthly']
        print(f"\n【月別成績（{label} RR逆算SL）】")
        print(f"  {'月':<8}{'全件':>5}{'全PF':>6}  {'買い件':>6}{'買PF':>6}  {'売り件':>6}{'売PF':>6}")
        print("  " + "-" * 52)
        for ym, g in monthly_r.items():
            a, b, s = g['all'], g['buy'], g['sell']
            def pf_s(v): return f"{v:.2f}" if v != float('inf') else " inf"
            print(f"  {ym:<8}{a['n']:>5}{pf_s(a['pf']):>6}  "
                  f"{b['n']:>6}{pf_s(b['pf']):>6}  "
                  f"{s['n']:>6}{pf_s(s['pf']):>6}")
        print("  " + "-" * 52)

    # ──── CSV出力 ────
    # 全バッファの明細を1ファイルに統合
    trade_csv = 'backtest_swing_trades.csv'
    with open(trade_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sl_buffer_mult','eval_dt','code','direction','entry','tp','sl','exit',
            'result','pnl_pct','rr_real','mfe_pct','mae_pct',
            'd1_ohlc_pct','d2_ohlc_pct','d3_ohlc_pct','d4_ohlc_pct','d5_ohlc_pct',
            'cumulative_ohlc','ma_divergence','trend_aligned','swing_rank',
        ])
        writer.writeheader()
        for mult in SL_BUFFER_MULTS:
            writer.writerows(all_trades[mult])
    print(f"\n全トレード明細（全バッファ統合）: {trade_csv}")

    # 比較サマリーCSV
    summary_csv = 'backtest_swing_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['指標'] + [f'SL×{m}' for m in SL_BUFFER_MULTS])
        for key, label in [
            ('win_rate',          '勝率(%)'),
            ('pf',                'プロフィットF'),
            ('tp_wins',           'TP到達'),
            ('f_wins',            '強制益'),
            ('sl_losses',         'SL到達'),
            ('f_losses',          '強制損'),
            ('avg_pnl',           '平均損益(%)'),
            ('avg_mfe',           '平均MFE(%)'),
            ('avg_mae',           '平均MAE(%)'),
            ('dir_rate',          '方向的中率(%)'),
            ('avg_ohlc_all',      'OHLC累積_全平均'),
            ('avg_ohlc_sl',       'OHLC累積_SL到達平均'),
            ('sl_ohlc_pos_rate',  'SL早すぎ疑い率(%)'),
            ('sl_ohlc_positive',  'SL到達でOHLCプラス件数'),
        ]:
            writer.writerow([label] + [all_metrics[m][key] for m in SL_BUFFER_MULTS])
    print(f"比較サマリー: {summary_csv}")

    # 方向×トレンド分析CSV
    trend_csv = 'backtest_swing_trend_analysis.csv'
    with open(trend_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['グループ', '件数', '勝率(%)', 'PF', 'SL率(%)', '平均損益(%)'])
        for label in ['買い×順張り', '買い×逆張り', '売り×順張り', '売り×逆張り']:
            g = dt_analysis[label]
            pf_val = g['pf'] if g['pf'] != float('inf') else 'inf'
            writer.writerow([label, g['n'], g['win_rate'], pf_val, g['sl_rate'], g['avg_pnl']])
    print(f"方向×トレンド分析: {trend_csv}")

    monthly_csv = 'backtest_swing_monthly.csv'
    with open(monthly_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['月', '全件数', '全勝率(%)', '全PF', '全平均損益(%)',
                         '買い件数', '買い勝率(%)', '買いPF', '買い平均損益(%)',
                         '売り件数', '売り勝率(%)', '売りPF', '売り平均損益(%)'])
        for ym, g in monthly.items():
            a, b, s = g['all'], g['buy'], g['sell']
            def pf_v(v): return v if v != float('inf') else 'inf'
            writer.writerow([ym,
                a['n'], a['win_rate'], pf_v(a['pf']), a['avg_pnl'],
                b['n'], b['win_rate'], pf_v(b['pf']), b['avg_pnl'],
                s['n'], s['win_rate'], pf_v(s['pf']), s['avg_pnl'],
            ])
    print(f"月別分析: {monthly_csv}")

    # 市場フィルターあり明細CSV（MA5/25版を保存）
    filtered_csv = 'backtest_swing_filtered.csv'
    trades_f = filter_results['MA5/25']['trades']
    with open(filtered_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sl_buffer_mult','eval_dt','code','direction','entry','tp','sl','exit',
            'result','pnl_pct','rr_real','mfe_pct','mae_pct',
            'd1_ohlc_pct','d2_ohlc_pct','d3_ohlc_pct','d4_ohlc_pct','d5_ohlc_pct',
            'cumulative_ohlc','ma_divergence','trend_aligned','swing_rank',
        ])
        writer.writeheader()
        writer.writerows(trades_f)
    print(f"市場フィルターあり明細: {filtered_csv}")

    # ──────────────────────────────────────────────────────────────────
    # [4/4] 指値バックテスト実行（RR目標別・エントリー待機あり）
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"[4/4] 指値バックテスト実行中...")
    print(f"  設定: SL_BUF={SL_BUF*100:.1f}%  TP_BUF={TP_BUF*100:.1f}%  "
          f"エントリー待機={ENTRY_WINDOW}日  ATRフィルター={ATR_FILTER_MULT}倍")
    print(f"{'='*72}")

    limit_results = {}
    for rr in RR_TARGETS:
        print(f"  RR={rr:.1f} 実行中...", end="\r")
        trades_l, stats_l = run_backtest_limit(
            all_data, eval_dates, rr_target=rr)
        limit_results[rr] = {
            'trades':  trades_l,
            'stats':   stats_l,
            'metrics': calc_metrics_limit(trades_l, stats_l),
        }
    print()

    # ── 比較表表示 ──
    print(f"\n{'='*80}")
    print("【指値バックテスト RR目標別比較】")
    print(f"  ※ 成行(×1.0)は参考値（エントリー率100%の従来方式）")
    print(f"{'='*80}")

    base_m = all_metrics[1.0]  # 成行ベースライン
    col_w  = 10
    header = (f"{'指標':<20}{'成行(参考)':>{col_w}}"
              + "".join(f"{'RR='+str(rr):>{col_w}}" for rr in RR_TARGETS))
    print(header)
    print("-" * 80)

    def lrow(label, base_val, vals, fmt=".1f"):
        bv   = format(base_val, fmt)
        line = "".join(format(v, fmt).rjust(col_w) for v in vals)
        print(f"{label:<20}{bv:>{col_w}}{line}")

    lrow("エントリー率(%)",  100.0,
         [limit_results[rr]['metrics'].get('entry_rate', 0) for rr in RR_TARGETS])
    lrow("勝率(%)",          base_m['win_rate'],
         [limit_results[rr]['metrics']['win_rate']  for rr in RR_TARGETS])
    lrow("プロフィットF",    base_m['pf'],
         [limit_results[rr]['metrics']['pf']        for rr in RR_TARGETS], fmt=".2f")
    lrow("SL到達率(%)",      round(base_m['sl_losses']/base_m['n']*100,1),
         [limit_results[rr]['metrics']['sl_rate']   for rr in RR_TARGETS])
    lrow("TP到達率(%)",      round(base_m['tp_wins']/base_m['n']*100,1),
         [limit_results[rr]['metrics']['tp_rate']   for rr in RR_TARGETS])
    lrow("平均損益(%)",      base_m['avg_pnl'],
         [limit_results[rr]['metrics']['avg_pnl']   for rr in RR_TARGETS], fmt="+.2f")
    lrow("平均MFE(%)",       base_m['avg_mfe'],
         [limit_results[rr]['metrics']['avg_mfe']   for rr in RR_TARGETS], fmt=".2f")
    lrow("平均MAE(%)",       base_m['avg_mae'],
         [limit_results[rr]['metrics']['avg_mae']   for rr in RR_TARGETS], fmt=".2f")
    print("-" * 80)

    # ── 日次含み益（MFE）推移 ──
    print(f"\n【日次含み益（MFE）推移 ― エントリー後N日目の最大含み益平均】")
    print(f"  成行(参考): OHLC累積スコアとは定義が異なるため省略")
    day_header = f"{'':>8}" + "".join(f"{'RR='+str(rr):>{col_w}}" for rr in RR_TARGETS)
    print(day_header)
    for d in range(HOLD_DAYS):
        row = f"  {d+1}日目MFE"
        for rr in RR_TARGETS:
            avgs = limit_results[rr]['metrics']['day_mfe_avgs']
            row += f"{avgs[d]:>{col_w}.3f}%"
        print(row)
    print()
    print(f"{'':>8}" + "".join(f"{'RR='+str(rr):>{col_w}}" for rr in RR_TARGETS))
    for d in range(HOLD_DAYS):
        row = f"  {d+1}日目終値"
        for rr in RR_TARGETS:
            avgs = limit_results[rr]['metrics']['day_ret_avgs']
            row += f"{avgs[d]:>{col_w}.3f}%"
        print(row)

    # ── CSV出力 ──
    # ──────────────────────────────────────────────────────────────────
    # [5/5] 変調検知バックテスト（敗者復活候補分析）
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"[5/5] 変調検知バックテスト実行中...")
    print(f"  閾値: ハイブリッドMA通過 < 10件 → 変調日として敗者復活候補を抽出")
    print(f"  敗者復活条件: 逆方向 × MA5/25のみ一致（MA25/75は不問）")
    HENSHO_THRESHOLD = 10

    if nikkei_data is not None:
        hensho_daily, hensho_trades = run_hensho_analysis(
            all_data, eval_dates, nikkei_data, threshold=HENSHO_THRESHOLD
        )

        hensho_days  = [d for d in hensho_daily if d['is_hensho']]
        normal_days  = [d for d in hensho_daily if not d['is_hensho']]

        print(f"\n{'='*72}")
        print(f"【変調検知サマリー】")
        print(f"{'='*72}")
        print(f"  評価日数計     : {len(hensho_daily)}日")
        print(f"  通常日(>=5件)  : {len(normal_days)}日  平均通過件数: "
              f"{sum(d['normal_count'] for d in normal_days)/len(normal_days):.1f}" if normal_days else "")
        print(f"  変調日(<5件)   : {len(hensho_days)}日")
        print()

        if hensho_days:
            print(f"  【変調日一覧】")
            print(f"  {'日付':<12}{'日経':<8}{'方向':<6}{'通常件数':>6}  {'5日後':>7}  {'敗者復活':>6}  {'復活PF':>7}")
            print(f"  {'-'*60}")
            for d in hensho_days:
                fwd  = f"{d['nikkei_5d_return']:+.1f}%" if d['nikkei_5d_return'] is not None else "  N/A"
                pf   = f"{d['repechage_pf']:.2f}"       if d['repechage_pf'] is not None      else "  N/A"
                print(f"  {d['date']:<12}{d['nikkei_close']:<8.0f}{d['nikkei_dir']:<6}"
                      f"{d['normal_count']:>6}  {fwd:>7}  {d['repechage_count']:>6}件  {pf:>7}")

        if hensho_trades:
            wins   = [t for t in hensho_trades if t['result'] in ('win', 'forced_win')]
            losses = [t for t in hensho_trades if t['result'] in ('loss', 'forced_loss')]
            gains  = sum(t['pnl_pct'] for t in wins)
            loss_sum = abs(sum(t['pnl_pct'] for t in losses))
            pf_rep = gains / loss_sum if loss_sum > 0 else float('inf')
            avg    = sum(t['pnl_pct'] for t in hensho_trades) / len(hensho_trades)
            print(f"\n  【敗者復活候補 総合成績】")
            print(f"  総トレード数  : {len(hensho_trades)}件")
            print(f"  勝率          : {len(wins)/len(hensho_trades)*100:.1f}%")
            print(f"  プロフィットF : {pf_rep:.2f}")
            print(f"  平均損益      : {avg:+.2f}%")

        # CSV出力
        hensho_daily_csv = 'backtest_hensho_daily.csv'
        with open(hensho_daily_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'date','nikkei_close','nikkei_5d_return','nikkei_dir',
                'normal_count','is_hensho','repechage_count','repechage_pf','repechage_avg_pnl'
            ])
            writer.writeheader()
            writer.writerows(hensho_daily)
        print(f"\n  変調日サマリー: {hensho_daily_csv}")

        if hensho_trades:
            hensho_trades_csv = 'backtest_hensho_trades.csv'
            with open(hensho_trades_csv, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(hensho_trades[0].keys()))
                writer.writeheader()
                writer.writerows(hensho_trades)
            print(f"  敗者復活明細  : {hensho_trades_csv}")
    else:
        print("  日経データ未取得のためスキップ")

    limit_csv = 'backtest_limit_trades.csv'
    all_limit_trades = []
    for rr in RR_TARGETS:
        all_limit_trades.extend(limit_results[rr]['trades'])

    if all_limit_trades:
        fieldnames = list(all_limit_trades[0].keys())
        with open(limit_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_limit_trades)
        print(f"\n指値バックテスト明細: {limit_csv}")
