"""
スイングスコア配点変更 バックテスト
OLD: トレンド35(相対ランク) + RSI30 + %B10 + RR15 + MACD10 = 100点
NEW: トレンド20(絶対評価)  + RSI35 + %B20 + RR15 + MACD10 = 100点

除外ルール
  ルール1: 決算7日以内の銘柄を除外
  ルール2: 前日比 ±5% 超の銘柄を除外
  ルール3: 買いで RSI>70 / 売りで RSI<30 を除外（信用倍率の代理指標）

評価指標: 5営業日後の方向的中率
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
# ユニバース（日経225 流動性上位30銘柄）
# ─────────────────────────────────────────────
UNIVERSE = [
    "7203","6758","9984","8306","6501","9432","4502","7011","6861","8035",
    "6367","4063","6902","6954","7974","9020","8031","3382","6857","4519",
    "8316","7267","4568","7751","8766","2914","6752","4503","9983","6273",
]

HOLD_DAYS = 5   # 保有日数（翌5営業日後で評価）
TOP_N     = 5   # 各期間のTop N銘柄で評価
HIST_BARS = 80  # スコア計算に必要な最低バー数

# ─────────────────────────────────────────────
# テクニカル指標（純粋な numpy/pandas 実装）
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
    e12 = _ema(arr, 12)   # length n-11
    e26 = _ema(arr, 26)   # length n-25
    # e12[i+14] aligns with e26[i] (both → arr[i+25])
    macd_full = e26 - e12[14:]                     # length n-25
    if len(macd_full) < 9:
        return macd_full, np.array([])
    signal = _ema(macd_full, 9)                    # length n-33
    return macd_full[8:], signal                   # aligned

def calc_indicators(df):
    """DataFrameから生値を計算して返す辞書。失敗時はNone。"""
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

        # RSI
        rsi = _rsi(closes)

        # MA
        ma25 = float(np.mean(closes[-25:])) if n >= 25 else float('nan')
        ma75 = float(np.mean(closes[-75:])) if n >= 75 else float('nan')
        if np.isnan(ma75) or ma75 == 0:
            ma_divergence = 0.0
        else:
            ma_divergence = (ma25 - ma75) / ma75 * 100

        # ATR
        atr_val = _atr(highs, lows, closes)

        # %B（ボリバン）
        pct_b = _bb_pctb(closes)

        # HVN（サポート/レジスタンス）
        sup_p, res_p = calc_hvn(df, curr)
        rr_buy = rr_sell = 0.0
        if sup_p and res_p and curr != sup_p:
            rr_buy  = abs(res_p - curr) / abs(curr - sup_p)
        if sup_p and res_p and curr != res_p:
            rr_sell = abs(curr - sup_p) / abs(res_p - curr)

        return {
            'curr':         curr,
            'change_pct':   change_pct,
            'rsi':          rsi,
            'ma_divergence': ma_divergence,
            'atr_val':      atr_val,
            'pct_b':        pct_b,
            'rr_buy_raw':   rr_buy,
            'rr_sell_raw':  rr_sell,
            'sup_price':    sup_p,
            'res_price':    res_p,
            '_closes':      closes,
            '_df':          df,
        }
    except Exception as e:
        return None


def calc_hvn(df, curr, bins=20):
    """HVN計算（直近1ヶ月×2重み）。(sup, res) を返す。"""
    try:
        closes  = df['Close'].values.astype(float)
        volumes = df['Volume'].values.astype(float)
        dates   = df['Date'] if 'Date' in df.columns else df.index

        cutoff = pd.Timestamp(dates.max()) - pd.Timedelta(days=30)
        weights = np.where(pd.to_datetime(dates) >= cutoff, 2.0, 1.0)
        wv = volumes * weights

        bin_result = pd.cut(pd.Series(closes), bins=bins, labels=False, retbins=True)
        pbins     = bin_result[0]
        bin_edges = bin_result[1]
        bcenters  = (bin_edges[:-1] + bin_edges[1:]) / 2

        vol_by_bin = pd.Series(wv).groupby(pbins).sum()
        top5  = vol_by_bin.nlargest(5).index
        hvn   = sorted([bcenters[int(i)] for i in top5 if int(i) < len(bcenters)])

        sup = max([p for p in hvn if p < curr], default=None)
        res = min([p for p in hvn if p > curr], default=None)
        return sup, res
    except Exception:
        return None, None


def calc_macd_cross(closes, direction):
    """直近3本以内のMACDクロスで10点。"""
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


def rank_score(values, higher_is_better=True, max_pts=30):
    n = len(values)
    if n <= 1:
        return [max_pts] * n
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


# ─────────────────────────────────────────────
# RSI スコアテーブル
# ─────────────────────────────────────────────

def rsi_score_old(rsi):
    """OLD: 最大30点"""
    if   40 <= rsi <= 60: return 30, 30
    elif 30 <= rsi <  40: return 24, 12
    elif 60 <  rsi <= 70: return 12, 24
    elif rsi <  30:       return 30,  0
    else:                 return  0, 30   # > 70


def rsi_score_new(rsi):
    """NEW: 最大35点（既存テーブルを35点上限にスケール: ×7/6）"""
    if   40 <= rsi <= 60: return 35, 35
    elif 30 <= rsi <  40: return 28, 14
    elif 60 <  rsi <= 70: return 14, 28
    elif rsi <  30:       return 35,  0
    else:                 return  0, 35   # > 70


# ─────────────────────────────────────────────
# スイングスコア計算（OLD / NEW）
# ─────────────────────────────────────────────

def score_swing_old(raws):
    """
    OLD: トレンド35(相対) + RSI30 + %B10 + RR15 + MACD10
    """
    n = len(raws)
    ma_divs  = [r['ma_divergence'] for r in raws]
    rr_buys  = [r['rr_buy_raw']    for r in raws]
    rr_sells = [r['rr_sell_raw']   for r in raws]

    buy_trend_sc  = rank_score(ma_divs,  True,  35)
    sell_trend_sc = rank_score(ma_divs,  False, 35)
    buy_rr_sc     = rank_score(rr_buys,  True,  15)
    sell_rr_sc    = rank_score(rr_sells, True,  15)

    for i, r in enumerate(raws):
        b_rsi, s_rsi = rsi_score_old(r['rsi'])
        pctb = r['pct_b']
        b_bb  = max(0, round((1 - pctb) * 10, 1)) if pctb is not None else 0
        s_bb  = max(0, round(pctb * 10,       1)) if pctb is not None else 0

        buy_tmp  = buy_trend_sc[i]  + b_rsi + buy_rr_sc[i]  + b_bb
        sell_tmp = sell_trend_sc[i] + s_rsi + sell_rr_sc[i] + s_bb
        tent = "買い" if buy_tmp >= sell_tmp else "売り"

        macd = calc_macd_cross(r['_closes'], tent)
        buy_score  = buy_tmp  + (macd if tent == "買い" else 0)
        sell_score = sell_tmp + (macd if tent == "売り" else 0)

        if buy_score >= sell_score:
            r['swing_direction'] = "買い"
            r['swing_score']     = round(buy_score, 1)
        else:
            r['swing_direction'] = "売り"
            r['swing_score']     = round(sell_score, 1)
        r['swing_rr_valid'] = (r['sup_price'] is not None and
                               r['res_price'] is not None and
                               max(r['rr_buy_raw'], r['rr_sell_raw']) > 0)


def score_swing_new(raws):
    """
    NEW: トレンド20(絶対) + RSI35 + %B20 + RR15 + MACD10
    """
    n = len(raws)
    rr_buys  = [r['rr_buy_raw']  for r in raws]
    rr_sells = [r['rr_sell_raw'] for r in raws]
    buy_rr_sc  = rank_score(rr_buys,  True, 15)
    sell_rr_sc = rank_score(rr_sells, True, 15)

    for i, r in enumerate(raws):
        # トレンド（絶対評価）
        if r['ma_divergence'] > 0:
            b_tr, s_tr = 20, 0
        else:
            b_tr, s_tr = 0, 20

        b_rsi, s_rsi = rsi_score_new(r['rsi'])

        pctb = r['pct_b']
        b_bb  = max(0, round((1 - pctb) * 20, 1)) if pctb is not None else 0
        s_bb  = max(0, round(pctb * 20,       1)) if pctb is not None else 0

        buy_tmp  = b_tr + b_rsi + buy_rr_sc[i]  + b_bb
        sell_tmp = s_tr + s_rsi + sell_rr_sc[i] + s_bb
        tent = "買い" if buy_tmp >= sell_tmp else "売り"

        macd = calc_macd_cross(r['_closes'], tent)
        buy_score  = buy_tmp  + (macd if tent == "買い" else 0)
        sell_score = sell_tmp + (macd if tent == "売り" else 0)

        if buy_score >= sell_score:
            r['swing_direction'] = "買い"
            r['swing_score']     = round(buy_score, 1)
        else:
            r['swing_direction'] = "売り"
            r['swing_score']     = round(sell_score, 1)
        r['swing_rr_valid'] = (r['sup_price'] is not None and
                               r['res_price'] is not None and
                               max(r['rr_buy_raw'], r['rr_sell_raw']) > 0)


# ─────────────────────────────────────────────
# 除外ルール
# ─────────────────────────────────────────────

def apply_exclusions(raws, rule1=False, rule2=False, rule3=False,
                     earnings_near: set = None):
    result = []
    for r in raws:
        # ルール1: 決算直前除外
        if rule1 and earnings_near and r.get('code') in earnings_near:
            continue
        # ルール2: 前日比 ±5% 超
        if rule2 and abs(r['change_pct']) > 5.0:
            continue
        # ルール3: RSI極値（買いでRSI>70 / 売りでRSI<30）
        if rule3:
            d = r.get('swing_direction', '')
            if d == "買い" and r['rsi'] > 70:
                continue
            if d == "売り" and r['rsi'] < 30:
                continue
        result.append(r)
    return result


# ─────────────────────────────────────────────
# Walk-Forward バックテスト本体
# ─────────────────────────────────────────────

def run_backtest(all_data_dict, eval_dates, scorer_fn,
                 rule1=False, rule2=False, rule3=False,
                 earnings_map: dict = None):
    """
    scorer_fn: score_swing_old または score_swing_new（raws リストを破壊的に更新）
    returns: (hit_rate_pct, total_samples, correct_samples)
    """
    correct = total = 0
    earnings_map = earnings_map or {}

    for eval_dt in eval_dates:
        raws = []
        for code, df_full in all_data_dict.items():
            df_slice = df_full[df_full.index.normalize() < pd.Timestamp(eval_dt).normalize()]
            if len(df_slice) < HIST_BARS:
                continue
            raw = calc_indicators(df_slice)
            if raw is None:
                continue
            raw['code'] = code
            raws.append(raw)

        if len(raws) < 3:
            continue

        # スコア計算
        import copy
        raws_copy = copy.deepcopy(raws)
        scorer_fn(raws_copy)

        # 除外ルール適用（ルール3はスコア計算後に実施）
        earnings_near = earnings_map.get(eval_dt, set())
        candidates = [r for r in raws_copy if r.get('swing_rr_valid', False)]
        candidates = apply_exclusions(candidates,
                                      rule1=rule1, rule2=rule2, rule3=rule3,
                                      earnings_near=earnings_near)
        if not candidates:
            continue

        top = sorted(candidates, key=lambda x: x['swing_score'], reverse=True)[:TOP_N]

        # 5営業日後リターン確認
        for s in top:
            code = s['code']
            df_full = all_data_dict[code]
            # 評価日の終値
            past = df_full[df_full.index.normalize() <= pd.Timestamp(eval_dt).normalize()]
            if past.empty:
                continue
            entry_price = float(past['Close'].iloc[-1])

            # 5営業日後
            future = df_full[df_full.index.normalize() > pd.Timestamp(eval_dt).normalize()].head(HOLD_DAYS)
            if len(future) < HOLD_DAYS:
                continue
            exit_price = float(future['Close'].iloc[-1])

            fwd_return = (exit_price - entry_price) / entry_price
            direction  = s['swing_direction']

            if direction == "買い" and fwd_return > 0:
                correct += 1
            elif direction == "売り" and fwd_return < 0:
                correct += 1
            total += 1

    hit_rate = correct / total * 100 if total > 0 else 0.0
    return round(hit_rate, 2), total, correct


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("スイングスコア配点変更 バックテスト")
    print("OLD: トレンド35(相対) + RSI30 + %B10 + RR15 + MACD10")
    print("NEW: トレンド20(絶対) + RSI35 + %B20 + RR15 + MACD10")
    print("=" * 60)

    # ──── データ取得 ────
    end_dt   = datetime.now() - timedelta(days=5)
    start_dt = end_dt - timedelta(days=400)   # ~13ヶ月

    print(f"\n[1/3] 株価データ取得中（{len(UNIVERSE)}銘柄）...")
    all_data = {}
    for code in UNIVERSE:
        try:
            df = yf.Ticker(f"{code}.T").history(
                start=start_dt.strftime('%Y-%m-%d'),
                end=(end_dt + timedelta(days=30)).strftime('%Y-%m-%d')
            )
            if len(df) >= HIST_BARS + HOLD_DAYS + 20:
                df.index = df.index.tz_localize(None)
                all_data[code] = df
                print(f"  {code}: {len(df)}本", end="\r")
        except Exception as e:
            print(f"  {code}: 取得失敗 ({e})")
        time.sleep(0.15)
    print(f"\n取得成功: {len(all_data)}銘柄")

    # ──── 評価日生成（過去4ヶ月・5営業日おき）────
    print("\n[2/3] 評価日生成中...")
    eval_end   = end_dt - timedelta(days=HOLD_DAYS + 5)
    eval_start = eval_end - timedelta(days=120)

    cal_df = next(iter(all_data.values()))
    trading_days = list(cal_df[(cal_df.index >= pd.Timestamp(eval_start)) &
                               (cal_df.index <= pd.Timestamp(eval_end))].index)
    eval_dates = [d.to_pydatetime() for d in trading_days[::5]]

    if not eval_dates:
        print("ERROR: 評価日が生成できませんでした")
        exit(1)

    print(f"  評価期間: {eval_dates[0].strftime('%Y-%m-%d')} ～ {eval_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  評価ポイント数: {len(eval_dates)}")

    # ──── 全パターン実行 ────
    print("\n[3/3] バックテスト実行中...")

    PATTERNS = [
        # (ラベル, scorer, rule1, rule2, rule3)
        ("OLD(従来)　　　 除外なし",       score_swing_old, False, False, False),
        ("OLD(従来)　　　 ルール1のみ",    score_swing_old, True,  False, False),
        ("OLD(従来)　　　 ルール2のみ",    score_swing_old, False, True,  False),
        ("OLD(従来)　　　 ルール3のみ",    score_swing_old, False, False, True),
        ("OLD(従来)　　　 ルール1+2",      score_swing_old, True,  True,  False),
        ("OLD(従来)　　　 ルール1+3",      score_swing_old, True,  False, True),
        ("OLD(従来)　　　 ルール2+3",      score_swing_old, False, True,  True),
        ("OLD(従来)　　　 全ルール",        score_swing_old, True,  True,  True),
        ("NEW(新配点)　　 除外なし",        score_swing_new, False, False, False),
        ("NEW(新配点)　　 ルール1のみ",     score_swing_new, True,  False, False),
        ("NEW(新配点)　　 ルール2のみ",     score_swing_new, False, True,  False),
        ("NEW(新配点)　　 ルール3のみ",     score_swing_new, False, False, True),
        ("NEW(新配点)　　 ルール1+2",       score_swing_new, True,  True,  False),
        ("NEW(新配点)　　 ルール1+3",       score_swing_new, True,  False, True),
        ("NEW(新配点)　　 ルール2+3",       score_swing_new, False, True,  True),
        ("NEW(新配点)　　 全ルール",         score_swing_new, True,  True,  True),
    ]

    rows = []
    for label, scorer, r1, r2, r3 in PATTERNS:
        pct, total, correct = run_backtest(
            all_data, eval_dates, scorer,
            rule1=r1, rule2=r2, rule3=r3
        )
        rows.append((label, pct, total, correct))
        print(f"  {label}: {pct:.1f}%  ({correct}/{total}サンプル)")

    # ──── CSV出力 ────
    out_file = 'backtest_swing_comparison.csv'
    with open(out_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['検証パターン', '的中率', 'サンプル数', '的中数'])
        writer.writerow(['─ OLD（配点変更前）─', '', '', ''])
        for label, pct, total, correct in rows[:8]:
            writer.writerow([label.strip(), f'{pct:.1f}%', total, correct])
        writer.writerow(['─ NEW（配点変更後）─', '', '', ''])
        for label, pct, total, correct in rows[8:]:
            writer.writerow([label.strip(), f'{pct:.1f}%', total, correct])

    print(f"\n結果を {out_file} に保存しました")

    # ──── サマリー表示 ────
    print("\n" + "=" * 60)
    print("【比較サマリー】除外なし ベースライン")
    print("=" * 60)
    old_base = next(r for r in rows if "OLD" in r[0] and "除外なし" in r[0])
    new_base = next(r for r in rows if "NEW" in r[0] and "除外なし" in r[0])
    print(f"  OLD 除外なし: {old_base[1]:.1f}%  ({old_base[3]}/{old_base[2]})")
    print(f"  NEW 除外なし: {new_base[1]:.1f}%  ({new_base[3]}/{new_base[2]})")
    diff = new_base[1] - old_base[1]
    sign = "+" if diff >= 0 else ""
    print(f"  差分 (NEW-OLD): {sign}{diff:.1f}pt")

    print("\n【注記】")
    print("  ルール1: 決算7日以内の銘柄を除外")
    print("  ルール2: 前日比 ±5% 超の銘柄を除外")
    print("  ルール3: 買いでRSI>70 / 売りでRSI<30 の銘柄を除外")
    print("           （信用倍率が取得不可のため RSI 極値を代理指標として使用）")
    print(f"  評価: {HOLD_DAYS}営業日後終値の方向的中率、Top{TOP_N}銘柄")
