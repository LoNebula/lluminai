# 📁 Gemini File Search Tool — PDF Summarization Demo

このリポジトリは、**Google Gemini API の File Search Tool** を使って
PDF をアップロードし、モデル自身に **検索・引用・要約** させる最小構成のデモです。

ブログ記事
**「Gemini apiのfile search tool を実際に触ってみた」**（）
で解説したサンプルコード一式をそのまま再現しています。

---

## 🚀 概要

Gemini の File Search Tool は、PDF や画像・音声などをアップロードすると、
AI が内部で自動インデックス化し、必要に応じて自律的に検索して回答してくれる仕組みです。

つまり **RAG を自分で構築せずに検索型エージェントを作れる** のがポイント。

このリポジトリでは次のことができます：

* PDF を Gemini File API にアップロード
* インデックス化された PDF をモデルが検索
* 要約内容の生成
* （モデルが引用したソース情報も返ってくる）

---

## 📦 ディレクトリ構成

```
.
├── main.py                # メイン実装（PDFアップロード & 要約）
├── 2511.14383v1.pdf       # サンプルPDF（数学論文）
└── README.md              # 本ファイル
```

---

## 🔧 動作環境

* Python 3.10+
* `google-genai` SDK
* `.env` に Google API Key を設定

---

## 🛠 セットアップ

### 1. 依存ライブラリのインストール

```bash
pip install google-genai python-dotenv
```

### 2. `.env` ファイルを作成

```
GOOGLE_API_KEY=あなたのAPIキー
```

### 3. 実行

```bash
python main.py
```

---

## 🧠 何が起きるのか？

`main.py` では以下の処理を行っています（）：

1. PDF（`2511.14383v1.pdf`）を File API にアップロード
2. アップロードされたファイル URI を取得
3. `generate_content()` に PDF URI を渡し、モデルへ質問
4. モデルが PDF 内を検索し、引用メタデータ付きで回答