import io
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="株式バックテスト", page_icon="📊", layout="wide")

st.title("📊 株式バックテストシステム")
st.caption("過去の日足データで、一目均衡表・RSIを使った売買ルールを検証します。売買注文は行いません。")

st.info(
    "初期版は日足専用です。シグナルは日足の終値確定後に判定し、翌営業日の始値でエントリーします。"
    "現在の銘柄リストを使う検証には、上場廃止銘柄が含まれないことによる生存者バイアスがあります。"
)

DEFAULT_TICKERS = """7203.T,6758.T,9984.T,8306.T,6861.T,AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA"""

with st.sidebar:
    st.header("検証条件")
    period_label = st.selectbox(
        "過去データ期間",
        ["2年", "5年", "10年"],
        index=1,
        help="Yahoo Financeで取得できる期間や銘柄によっては、全期間のデータが揃わない場合があります。",
    )
    period_map = {"2年": "2y", "5年": "5y", "10年": "10y"}
    period = period_map[period_label]

    stop_pct = st.number_input("損切り（%）", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
    target_pct = st.number_input("利確（%）", min_value=0.5, max_value=100.0, value=10.0, step=0.5)
    max_hold = st.number_input("最大保有日数（営業日）", min_value=1, max_value=120, value=20, step=1)
    fee_pct = st.number_input("片道コスト（手数料＋スリッページ %）", min_value=0.0, max_value=2.0, value=0.10, step=0.01)
    st.caption("片道コストは売買それぞれに適用します。例：0.10%なら往復約0.20%。")

    st.divider()
    st.subheader("売買シグナル")
    rsi_min = st.number_input("RSI下限", min_value=1, max_value=99, value=50)
    rsi_max = st.number_input("RSI上限", min_value=1, max_value=100, value=70)
    require_cloud = st.checkbox("終値が一目均衡表の雲より上", value=True)
    require_cross = st.checkbox("転換線が基準線より上", value=True)

st.subheader("1. 検証する銘柄")
st.write("銘柄コードをカンマまたは改行で入力するか、CSVをアップロードしてください。CSVは `ticker` 列が必要です。")
ticker_text = st.text_area("銘柄コード", value=DEFAULT_TICKERS, height=100)
uploaded = st.file_uploader("銘柄CSV（任意）", type=["csv"])

tickers = []
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        if "ticker" not in uploaded_df.columns:
            st.error("CSVに ticker 列がありません。例：ticker,market")
        else:
            tickers.extend(uploaded_df["ticker"].dropna().astype(str).tolist())
    except Exception as exc:
        st.error(f"CSVを読み込めませんでした：{exc}")

if ticker_text.strip():
    tickers.extend([x.strip() for x in ticker_text.replace("\n", ",").split(",") if x.strip()])

# Keep input order while removing duplicates.
tickers = list(dict.fromkeys(tickers))
st.caption(f"入力銘柄数：{len(tickers)}")

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators using only data available at each daily close."""
    x = df.copy()
    high = x["High"]
    low = x["Low"]
    close = x["Close"]

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a_raw = (tenkan + kijun) / 2
    span_b_raw = (high.rolling(52).max() + low.rolling(52).min()) / 2

    # Cloud visible on date t was calculated 26 sessions earlier.
    x["tenkan"] = tenkan
    x["kijun"] = kijun
    x["cloud_a"] = span_a_raw.shift(26)
    x["cloud_b"] = span_b_raw.shift(26)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    x.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100
    x.loc[(avg_loss == 0) & (avg_gain == 0), "rsi"] = 50
    return x

def simulate_one(ticker: str, raw: pd.DataFrame):
    if raw is None or raw.empty:
        return [], "価格データなし"

    # yfinance multi-ticker downloads may return a MultiIndex.
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(-1):
            raw = raw.xs(ticker, axis=1, level=-1)
        elif ticker in raw.columns.get_level_values(0):
            raw = raw.xs(ticker, axis=1, level=0)
        else:
            return [], "取得データの列を特定できません"

    needed = ["Open", "High", "Low", "Close"]
    if not all(c in raw.columns for c in needed):
        return [], "OHLCデータ不足"

    data = raw[needed].copy().dropna()
    if len(data) < 100:
        return [], "データ不足（100営業日未満）"

    data = calc_indicators(data)
    cloud_top = data[["cloud_a", "cloud_b"]].max(axis=1)
    signal = data["rsi"].between(rsi_min, rsi_max)
    if require_cloud:
        signal &= data["Close"] > cloud_top
    if require_cross:
        signal &= data["tenkan"] > data["kijun"]
    signal &= data[["rsi", "tenkan", "kijun"]].notna().all(axis=1)
    if require_cloud:
        signal &= cloud_top.notna()

    trades = []
    i = 0
    dates = data.index
    while i < len(data) - 1:
        if not bool(signal.iloc[i]):
            i += 1
            continue

        entry_i = i + 1
        entry_date = dates[entry_i]
        entry_open = float(data["Open"].iloc[entry_i])
        if not np.isfinite(entry_open) or entry_open <= 0:
            i += 1
            continue

        # Entry slippage/cost is applied to the purchase price.
        entry_price = entry_open * (1 + fee_pct / 100)
        stop_price = entry_price * (1 - stop_pct / 100)
        target_price = entry_price * (1 + target_pct / 100)
        last_i = min(entry_i + int(max_hold) - 1, len(data) - 1)

        exit_i = last_i
        exit_price_raw = float(data["Close"].iloc[last_i])
        exit_reason = "最大保有日数"

        for j in range(entry_i, last_i + 1):
            day_open = float(data["Open"].iloc[j])
            day_low = float(data["Low"].iloc[j])
            day_high = float(data["High"].iloc[j])

            # Conservative handling: if both levels are touched in one daily bar,
            # assume the stop is hit first. Gap-through exits use the opening price.
            if day_open <= stop_price:
                exit_i, exit_price_raw, exit_reason = j, day_open, "損切り（窓開け）"
                break
            if day_open >= target_price:
                exit_i, exit_price_raw, exit_reason = j, day_open, "利確（窓開け）"
                break
            stop_hit = day_low <= stop_price
            target_hit = day_high >= target_price
            if stop_hit:
                exit_i, exit_price_raw, exit_reason = j, stop_price, "損切り"
                break
            if target_hit:
                exit_i, exit_price_raw, exit_reason = j, target_price, "利確"
                break

        exit_price = exit_price_raw * (1 - fee_pct / 100)
        ret_pct = (exit_price / entry_price - 1) * 100
        trades.append({
            "ticker": ticker,
            "entry_date": pd.Timestamp(entry_date).date().isoformat(),
            "exit_date": pd.Timestamp(dates[exit_i]).date().isoformat(),
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "return_pct": round(ret_pct, 3),
            "holding_days": int(exit_i - entry_i + 1),
            "exit_reason": exit_reason,
        })
        # No overlapping positions in the same ticker.
        i = exit_i + 1

    return trades, None

run = st.button("バックテストを実行", type="primary", disabled=(len(tickers) == 0))
if run:
    with st.spinner("過去株価データを取得して検証しています。銘柄数によって時間がかかります…"):
        try:
            # Batch downloads reduce request count and help with larger universes.
            all_trades = []
            errors = []
            progress = st.progress(0)
            status = st.empty()
            batch_size = 40
            batches = [tickers[k:k + batch_size] for k in range(0, len(tickers), batch_size)]

            for b_idx, batch in enumerate(batches):
                status.write(f"データ取得中：{min((b_idx + 1) * batch_size, len(tickers))}/{len(tickers)} 銘柄")
                try:
                    downloaded = yf.download(
                        tickers=batch,
                        period=period,
                        interval="1d",
                        auto_adjust=True,
                        group_by="ticker",
                        threads=True,
                        progress=False,
                        timeout=30,
                    )
                except Exception as exc:
                    downloaded = pd.DataFrame()
                    errors.extend([{"ticker": t, "error": f"取得エラー：{exc}"} for t in batch])

                for ticker in batch:
                    try:
                        if downloaded.empty:
                            frame = pd.DataFrame()
                        elif len(batch) == 1:
                            frame = downloaded
                        elif isinstance(downloaded.columns, pd.MultiIndex):
                            if ticker in downloaded.columns.get_level_values(0):
                                frame = downloaded[ticker]
                            elif ticker in downloaded.columns.get_level_values(-1):
                                frame = downloaded.xs(ticker, axis=1, level=-1)
                            else:
                                frame = pd.DataFrame()
                        else:
                            frame = pd.DataFrame()

                        trades, err = simulate_one(ticker, frame)
                        all_trades.extend(trades)
                        if err:
                            errors.append({"ticker": ticker, "error": err})
                    except Exception as exc:
                        errors.append({"ticker": ticker, "error": str(exc)})

                progress.progress((b_idx + 1) / len(batches))
                time.sleep(0.3)

            status.success("検証が完了しました。")
            result = pd.DataFrame(all_trades)
            error_df = pd.DataFrame(errors)

            st.session_state["bt_result"] = result
            st.session_state["bt_errors"] = error_df
        except Exception as exc:
            st.error(f"処理中にエラーが発生しました：{exc}")

if "bt_result" in st.session_state:
    result = st.session_state["bt_result"]
    errors = st.session_state.get("bt_errors", pd.DataFrame())

    st.divider()
    st.subheader("2. 検証結果")
    if result.empty:
        st.warning("売買が発生しませんでした。条件を少し緩めるか、銘柄コードを確認してください。")
    else:
        wins = result[result["return_pct"] > 0]
        losses = result[result["return_pct"] <= 0]
        win_rate = len(wins) / len(result) * 100
        avg_return = result["return_pct"].mean()
        total_return = ((1 + result["return_pct"] / 100).prod() - 1) * 100
        equity = (1 + result["return_pct"] / 100).cumprod()
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("取引回数", f"{len(result):,}回")
        c2.metric("勝率", f"{win_rate:.1f}%")
        c3.metric("平均損益率", f"{avg_return:.2f}%")
        c4.metric("最大ドローダウン", f"{max_dd:.2f}%")

        st.metric("複利換算の累積損益率（取引を連続再投資した仮定）", f"{total_return:.2f}%")
        st.caption("この累積損益率は全取引に資金を連続再投資した単純モデルです。同時保有・資金配分・税金は未反映です。")

        by_exit = result["exit_reason"].value_counts().rename_axis("決済理由").reset_index(name="回数")
        st.write("決済理由")
        st.dataframe(by_exit, use_container_width=True, hide_index=True)

        st.write("取引一覧")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "取引結果CSVをダウンロード",
            data=result.to_csv(index=False).encode("utf-8-sig"),
            file_name="backtest_trades.csv",
            mime="text/csv",
        )

    if not errors.empty:
        with st.expander(f"取得できなかった銘柄・エラー（{len(errors)}件）"):
            st.dataframe(errors, use_container_width=True, hide_index=True)
            st.download_button(
                "エラー一覧CSVをダウンロード",
                data=errors.to_csv(index=False).encode("utf-8-sig"),
                file_name="backtest_errors.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "注意：これは検証用の簡易モデルです。配当、税金、売買単位、為替、流動性、信用取引コストは未反映です。"
    "同一日中に損切り・利確の両方へ到達した場合は、保守的に損切りを先に約定したものとして扱います。"
)
