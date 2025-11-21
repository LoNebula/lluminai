# 💬 Sakura AI × Streamlit  
## 国内LLM（GPT-OSS 120B）で動く “チャット Web UI”

このリポジトリは、  
**さくらインターネットの Sakura AI（GPT-OSS 120B）を利用した、国内完結チャットアプリ Web UI**  
の実装です。

Streamlit をフロントエンドとして利用し、  
シンプルかつ使いやすい対話インタフェースを構築しています。

Sakura AI は **OpenAI互換 API** を提供しており、  
完全国内環境・安全性・そして **無料枠の多さ** により、  
PoC・研究室・企業内検証に非常に適しています。

---

# 🧩 概要（Overview）

本アプリケーションでは、Sakura AI の GPT-OSS モデルを利用し、

- 💬 **自然な対話を実現するチャットUI**
- 🔐 **APIトークンを .env で安全管理**
- 🎨 **Streamlit による軽量 Web UI**
- 🌸 **OpenAI互換のためコードがシンプル**
- 🆓 **無料枠が非常に大きく、多数の試行が可能**

を実現しています。

特に「国内クラウドで生成AIを使いたい」というニーズに強く、  
社内ツールの原型としてそのまま提案できるレベルの構造です。

---

# ⚙️ 使用環境（Environment）

| 項目 | 内容 |
|------|------|
| **モデル** | GPT-OSS-120B（Sakura AI） |
| **フレームワーク** | Streamlit, Python 3.10+ |
| **API** | Sakura AI Chat Completions（OpenAI互換） |
| **トークン管理** | python-dotenv |

---

# 📂 ディレクトリ構成（Directory Structure）

```

project_root/
│
├── app.py              # Streamlit Web UI（チャット画面）
├── sakura_client.py    # Sakura AI API クライアント
│
├── .env                # APIトークン
└── README.md

```

---

# 🚀 セットアップ（Setup）

## 1. Python環境の準備

```bash
pip install streamlit requests python-dotenv
```

---

## 2. Sakura AI の API トークンを設定

`.env` を作成：

```env
SAKURA_TOKEN=あなたのAPIキー
```

---

## 3. 動作確認

```bash
streamlit run app.py
```

ログ例：

```
▶ Opening browser...
💬 Sakura AI × Streamlit Chat App Started
```

---

# ▶️ 実行方法（Run）

ブラウザが開いたら、
以下にアクセスしてチャットを開始できます。

```
http://localhost:8501
```

送信欄に質問を入力すると、
Sakura AI が回答を返してきます。

---

# 📦 主要ファイルの説明（Main Files）

## `app.py`

* Streamlit による Web チャット UI
* 会話履歴管理（session_state）
* 新しいメッセージを上に表示（降順UI）
* 「履歴クリア」ボタンで初期化

---

## `sakura_client.py`

* Sakura AI の ChatCompletion API を呼び出すラッパー
* OpenAI API 互換形式でリクエスト送信
* レスポンスから assistant のメッセージを抽出

---

# ⭐ もし役に立ったら

ぜひ GitHub のスター 🌟 をお願いします！

生成AI・LLM・Streamlit系アプリを継続的に公開していきます。
