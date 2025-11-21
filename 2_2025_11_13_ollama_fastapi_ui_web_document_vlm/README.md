# 🖼️ Ollama × FastAPI × LLaVA  
## ローカルで動く「画像 → Markdown」ドキュメント解析 Web UI

このリポジトリは、  
**ローカル VLM（Vision-Language Model）を使って、画像ドキュメントを Markdown 構造へ自動変換する Web UI アプリケーション**  
の実装です。

Ollama をバックエンドに、FastAPI で Web UI を作り、  
スキャン画像・ホワイトボード写真・PDFのキャプチャなどを  
**クラウドに送らずローカルだけで解析可能** です。

---

# 🧩 概要（Overview）

本アプリケーションでは、Ollama で動作する LLaVA 系 VLM を用い、

- 📄 **ドキュメント画像の内容理解**
- 📝 **Markdown 形式への構造化（見出し・箇条書き・表など）**
- 🌐 **FastAPI による Web UI でのインタラクティブ操作**
- ⚡ **GPU が利用可能な環境では自動で高速推論**

を実現します。

特に、**研究室・企業内での機密文書解析** や、  
**クラウドAPIが使えない環境での PoC** に非常に適しています。

---

# ⚙️ 使用環境（Environment）

| 項目 | 内容 |
|------|------|
| **モデル** | `llava:13b`（Ollamaローカル実行） |
| **フレームワーク** | FastAPI, Python 3.10+, HTML/JS |
| **推論エンジン** | Ollama（ローカルLLM/VLM） |
| **GPU対応** | CUDA / Apple Silicon 自動検出 |

---

# 📂 ディレクトリ構成（Directory Structure）

```

project_root/
│
├── app.py                # FastAPI メインサーバ
├── ollama_client.py      # Ollama API（画像 → Markdown）
│
├── static/
│   └── index.html        # Web UI（画像アップロード・結果表示）
│
└── README.md

````

---

# 🚀 セットアップ（Setup）

## 1. Python環境の準備

```bash
pip install fastapi uvicorn requests python-multipart
````

## 2. Ollamaのインストール

公式サイトからダウンロード
👉 [https://ollama.com/](https://ollama.com/)

## 3. VLMモデルの取得

```bash
ollama pull llava:13b
```

---

# ▶️ 実行方法（Run）

## 1. Ollamaサーバを立ち上げる

```bash
ollama serve
```

ログ例：

```
Listening on 127.0.0.1:11434
CUDA backend loaded ...
```

## 2. FastAPI を起動

```bash
uvicorn app:app --reload
```

起動ログ例：

```
🚀 FastAPI started
💡 Ollama is using: GPU mode
```

## 3. Web UI にアクセス

ブラウザで開く：

```
http://127.0.0.1:8000
```
---

# 📦 主要ファイルの説明（Main Files）

## `app.py`

* FastAPI サーバ本体
* `/api/analyze` : 画像 → Markdown（Ollama推論）
* `/` : Web UI

## `ollama_client.py`

* 画像の Base64 化
* LLaVA へのプロンプト送信
* Markdown 出力処理

## `static/index.html`

* ブラウザ UI（アップロード + 結果表示 + コピー機能つき）

---

# 🔬 背景技術：VLMの内部構造

本アプリで使用する LLaVA 系 VLM の内部：

1. **Vision Encoder（CLIP）**
2. **Projection Layer**
3. **LLM（Vicuna / Mistral 系）**

画像 → 視覚トークン → 言語トークン → Markdown 生成
という流れで動作します。

---

# ⭐ もし役に立ったら

ぜひ GitHub のスター 🌟 をお願いします！