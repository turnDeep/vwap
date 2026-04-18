# TQQQ / SQQQ Intraday VWAP Trading System

[English](./README.md) | 日本語

このリポジトリは、TQQQ および SQQQ を対象とした 1分足 VWAP ＋ ATR Trailing Stop のデイトレード自動運用システムです。機械学習ベースのマルチシグナルエンジン（Stallion）を解体し、最適化された最速のシングルアルゴリズムに刷新されました。現金口座（Cash Account）特有の差金決済ルール（GFVルール）を回避するための安全ロック（1日1回までの買い制限）を標準搭載しています。

## 概要

- **対象銘柄**: TQQQ (ロング), SQQQ (ショート代替用インバース)
- **戦略**: 1分足 VWAP Crossover + ATR Trailing Stop
- **最適化パラメータ**: `ATR Period = 9`, `ATR Multiplier = 27.15`, `VWAP Threshold = 0.063%`
- **取引制限**: 現金口座ルール対応。1日につき「TQQQ」1スロット、「SQQQ」1スロットまでの買いしか実行しません（往復ビンタ・GFVペナルティの完全排除）。
- **完全自動クローズ**: 15:58（NY時間）に保有ポジションを強制全決済し、オーバーナイトリスクをゼロにします。

## 動作モード

`.env` の設定に依存します。

- `LIVE` (本番環境):
  - `WEBULL_APP_KEY`, `WEBULL_APP_SECRET`, `WEBULL_ACCOUNT_ID` がすべて設定されている場合。
- `DEMO` (デモ環境):
  - 上記のAPIキーが1つでも欠けている場合は、自動的にデモモードで起動し、実際の発注は行わずにDiscord通知のみを実行します。

## 取引ロジック (How It Works)

1. **データ収集**: FMP (Financial Modeling Prep) API から1分ごとに TQQQ / SQQQ のリアルタイム株価データを取得。
2. **高速計算**: `QuoteBarAggregator` にて 1分足の Typical Price / Cumulative Volume を集約し、Numba を活用した超高速エンジンで VWAP と ATR を計算。
3. **シグナル判定**:
   - TQQQ の終値が `VWAP * (1 + 0.063%)` を上抜けた場合 -> TQQQ を成行買い (上昇トレンド)
   - TQQQ の終値が `VWAP * (1 - 0.063%)` を下抜けた場合 -> SQQQ を成行買い (下落トレンド)
4. **資金管理 & ロック機構**:
   - TQQQ/SQQQ のいずれかの買いシグナルが出た際、すでに逆の銘柄を保有していれば、決済してから対象を購入します（ドテン売買）。
   - その日すでにその銘柄の「買い」を実行していた場合、GFV（Good Faith Violation）防止のため、シグナルを無視して現金待機（Flat）します。
5. **EODフラット**: NY時間の 15:58 に強制全決済が行われます。

## Discord 通知

`DISCORD_BOT_TOKEN` と `DISCORD_CHANNEL_ID` を設定することで、リアルタイム通知が有効化されます。

- プレマーケット状態の通知
- 買い発注の送信
- オーダーの失敗 / スキップの警告
- 市場クローズ時のサマリー（運用成績）

## 実行方法

```bash
# ボットの起動 (これ一つでループ稼働します)
python master_scheduler.py
```

## Docker 運用

```bash
docker compose up -d --build
```
