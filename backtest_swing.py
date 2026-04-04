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

HOLD_DAYS = 5
TOP_N     = 10
TOP_VOL   = 50
HIST_BARS = 80

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


def calc_hvn(df, curr, bins=20, atr=None, atr_mult=2.0, fallback_mult=3.0):
    """
    atr指定時: curr±ATR×atr_mult 以上離れたHVNを選択。
              該当HVNなし→ curr±ATR×fallback_mult をフォールバック目標とする。
    """
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
    except Exception:
        return None, None


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

        ma25 = float(np.mean(closes[-25:])) if n >= 25 else float('nan')
        ma75 = float(np.mean(closes[-75:])) if n >= 75 else float('nan')
        if np.isnan(ma75) or ma75 == 0:
            ma_divergence = 0.0
        else:
            ma_divergence = (ma25 - ma75) / ma75 * 100

        atr_val = _atr(highs, lows, closes)
        pct_b   = _bb_pctb(closes)

        # HVN（ATRベース損切・HVNベース利確）
        sup_p, res_p = calc_hvn(df, curr, atr=atr_val)
        rr_buy = rr_sell = 0.0
        if res_p is not None and atr_val > 0:
            rr_buy  = abs(res_p - curr) / atr_val
        if sup_p is not None and atr_val > 0:
            rr_sell = abs(curr - sup_p) / atr_val

        return {
            'curr':          curr,
            'change_pct':    change_pct,
            'rsi':           rsi,
            'ma_divergence': ma_divergence,
            'atr_val':       atr_val,
            'pct_b':         pct_b,
            'rr_buy_raw':    rr_buy,
            'rr_sell_raw':   rr_sell,
            'sup_price':     sup_p,
            'res_price':     res_p,
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
    if rr is None or rr <= 0: return 0
    if rr >= 3.0: return 15
    if rr >= 2.0: return 12
    if rr >= 1.5: return 9
    if rr >= 1.0: return 6
    return 3


# ─────────────────────────────────────────────
# スイングスコア計算（NEW配点）
# ─────────────────────────────────────────────

def score_swing_new(raws):
    """NEW: トレンド20(絶対) + RSI35 + %B20 + RR15(絶対) + MACD10"""
    for r in raws:
        b_tr, s_tr = (20, 0) if r['ma_divergence'] > 0 else (0, 20)
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

        atr = r['atr_val']
        curr = r['curr']
        if buy_score >= sell_score:
            r['swing_direction'] = "買い"
            r['swing_score']     = round(buy_score, 1)
            r['swing_tp']        = r['res_price']       # 利確 = HVN上値
            r['swing_sl']        = curr - atr            # 損切 = curr - ATR
            r['swing_rr']        = r['rr_buy_raw']
        else:
            r['swing_direction'] = "売り"
            r['swing_score']     = round(sell_score, 1)
            r['swing_tp']        = r['sup_price']        # 利確 = HVN下値
            r['swing_sl']        = curr + atr            # 損切 = curr + ATR
            r['swing_rr']        = r['rr_sell_raw']
        r['swing_rr_valid'] = (r['swing_tp'] is not None and atr > 0)


# ─────────────────────────────────────────────
# 除外ルール（ルール2のみ：前日比±3%超除外）
# ─────────────────────────────────────────────

def apply_exclusions(raws, rule2_threshold=3.0):
    return [r for r in raws if abs(r['change_pct']) <= rule2_threshold]


# ─────────────────────────────────────────────
# バックテスト本体（本格版）
# ─────────────────────────────────────────────

def run_backtest(all_data_dict, eval_dates):
    """
    本格版バックテスト：翌日寄り付きエントリー、TP/SL判定、強制決済
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
        candidates = apply_exclusions(candidates)
        if not candidates:
            continue

        top = sorted(candidates, key=lambda x: x['swing_score'], reverse=True)[:TOP_N]

        for s in top:
            code     = s['code']
            df_full  = all_data_dict[code]
            direction = s['swing_direction']
            tp_level  = s['swing_tp']
            sl_level  = s['swing_sl']

            # 翌営業日以降の棒足を取得
            future_bars = df_full[df_full.index.normalize() > eval_ts]
            if len(future_bars) < 1:
                continue

            # エントリー = 翌営業日の寄り付き
            entry_bar   = future_bars.iloc[0]
            entry_price = float(entry_bar['Open'])

            # 方向的中率用の5日後終値
            if len(future_bars) >= HOLD_DAYS:
                exit_close = float(future_bars.iloc[HOLD_DAYS - 1]['Close'])
                fwd_return = (exit_close - entry_price) / entry_price
                hit_dir = (direction == "買い" and fwd_return > 0) or \
                          (direction == "売り" and fwd_return < 0)
                direction_records.append(hit_dir)

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
                result = 'forced'
                exit_price = float(hold_bars.iloc[-1]['Close'])

            if direction == "買い":
                pnl     = exit_price - entry_price
                rr_real = pnl / abs(sl_level - entry_price) if sl_level != entry_price else 0.0
            else:
                pnl     = entry_price - exit_price
                rr_real = pnl / abs(sl_level - entry_price) if sl_level != entry_price else 0.0

            pnl_pct = pnl / entry_price * 100

            trade_records.append({
                'eval_dt':   eval_dt.strftime('%Y-%m-%d'),
                'code':      code,
                'direction': direction,
                'entry':     round(entry_price, 1),
                'tp':        round(tp_level, 1),
                'sl':        round(sl_level, 1),
                'exit':      round(exit_price, 1),
                'result':    result,
                'pnl_pct':   round(pnl_pct, 2),
                'rr_real':   round(rr_real, 2),
                'mfe_pct':   round(mfe / entry_price * 100, 2),
                'mae_pct':   round(mae / entry_price * 100, 2),
            })

    return trade_records, direction_records


def calc_metrics(trade_records, direction_records):
    if not trade_records:
        return {}

    wins   = [t for t in trade_records if t['result'] == 'win']
    losses = [t for t in trade_records if t['result'] == 'loss']
    forced = [t for t in trade_records if t['result'] == 'forced']
    n      = len(trade_records)

    win_rate     = len(wins) / n * 100
    gross_profit = sum(t['pnl_pct'] for t in wins)  + \
                   sum(t['pnl_pct'] for t in [f for f in forced if f['pnl_pct'] > 0])
    gross_loss   = abs(sum(t['pnl_pct'] for t in losses)) + \
                   abs(sum(t['pnl_pct'] for t in [f for f in forced if f['pnl_pct'] < 0]))
    pf           = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_rr       = np.mean([t['rr_real'] for t in trade_records])
    avg_mfe      = np.mean([t['mfe_pct'] for t in trade_records])
    avg_mae      = np.mean([t['mae_pct'] for t in trade_records])
    avg_pnl      = np.mean([t['pnl_pct'] for t in trade_records])

    dir_rate = sum(direction_records) / len(direction_records) * 100 if direction_records else 0.0

    return {
        'n':          n,
        'wins':       len(wins),
        'losses':     len(losses),
        'forced':     len(forced),
        'win_rate':   round(win_rate, 1),
        'pf':         round(pf, 2),
        'avg_rr':     round(avg_rr, 2),
        'avg_mfe':    round(avg_mfe, 2),
        'avg_mae':    round(avg_mae, 2),
        'avg_pnl':    round(avg_pnl, 2),
        'dir_rate':   round(dir_rate, 1),
        'dir_n':      len(direction_records),
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

    # ──── バックテスト実行 ────
    print("\n[3/3] バックテスト実行中...")
    trade_records, direction_records = run_backtest(all_data, eval_dates)

    metrics = calc_metrics(trade_records, direction_records)

    # ──── 結果表示 ────
    print("\n" + "=" * 60)
    print("【バックテスト結果】")
    print("=" * 60)
    print(f"  総トレード数  : {metrics['n']}")
    print(f"  勝ち(TP到達)  : {metrics['wins']}  負け(SL到達): {metrics['losses']}  強制決済: {metrics['forced']}")
    print(f"  勝率          : {metrics['win_rate']:.1f}%")
    print(f"  プロフィットF : {metrics['pf']:.2f}")
    print(f"  平均RR(実績)  : {metrics['avg_rr']:.2f}")
    print(f"  平均損益      : {metrics['avg_pnl']:.2f}%")
    print(f"  平均MFE       : {metrics['avg_mfe']:.2f}%")
    print(f"  平均MAE       : {metrics['avg_mae']:.2f}%")
    print(f"  ---")
    print(f"  方向的中率(5日後終値): {metrics['dir_rate']:.1f}%  ({metrics['dir_n']}件)")

    # ──── CSV出力 ────
    trade_csv = 'backtest_swing_trades.csv'
    with open(trade_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'eval_dt','code','direction','entry','tp','sl','exit',
            'result','pnl_pct','rr_real','mfe_pct','mae_pct'
        ])
        writer.writeheader()
        writer.writerows(trade_records)
    print(f"\n全トレード明細: {trade_csv}")

    summary_csv = 'backtest_swing_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['指標', '値'])
        writer.writerow(['総トレード数',         metrics['n']])
        writer.writerow(['勝ち(TP到達)',          metrics['wins']])
        writer.writerow(['負け(SL到達)',          metrics['losses']])
        writer.writerow(['強制決済',              metrics['forced']])
        writer.writerow(['勝率',                  f"{metrics['win_rate']:.1f}%"])
        writer.writerow(['プロフィットファクター', f"{metrics['pf']:.2f}"])
        writer.writerow(['平均RR(実績)',           f"{metrics['avg_rr']:.2f}"])
        writer.writerow(['平均損益',              f"{metrics['avg_pnl']:.2f}%"])
        writer.writerow(['平均MFE',               f"{metrics['avg_mfe']:.2f}%"])
        writer.writerow(['平均MAE',               f"{metrics['avg_mae']:.2f}%"])
        writer.writerow(['方向的中率(5日後終値)',  f"{metrics['dir_rate']:.1f}%"])
        writer.writerow(['方向的中率サンプル数',   metrics['dir_n']])
    print(f"サマリー: {summary_csv}")
