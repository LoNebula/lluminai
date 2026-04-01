# 🧠 Claude Code Agentic Workflow: FastAPI 完全自律構築ハンズオン

[![Zenn Article](https://img.shields.io/badge/Zenn-Read_Article-blue?logo=zenn)](https://zenn.dev/) [![Lluminai](https://img.shields.io/badge/Lluminai-R&D-purple)](#)

ルミナイR&Dチームの宮脇彰梧です。🦋
このリポジトリは、Anthropicの **Claude Code** が持つ「Agentic Workflow（自律型ワークフロー）」の威力を検証するため、**空のディレクトリからAIにすべて丸投げして構築させたFastAPIプロジェクト**です。

単なる「コード生成」ではなく、AI自身が計画を立て、環境を構築し、テストを実行し、**発生したエラーを自律的にデバッグして解決する**までのプロセスを体験・確認することができます。

## 💡 本プロジェクトの見どころ（AIの自律的リカバリ）

通常、LLMにインメモリのSQLiteとSQLAlchemyを使ったテストコードを書かせると、セッションごとの接続分離仕様によりエラー（テーブルが見つからない等）が発生しがちです。

本プロジェクト構築時、Claude Codeは `pytest` 実行時の `Exit code 1` という標準エラー出力を自ら読み取り、以下のように**自己推論と自己修正**を行いました。

> **Claude Codeの推論ログ:**
> "The issue is SQLite in-memory DB: each connection gets its own isolated database. create_all uses one connection, but session requests use another. Fix: use StaticPool to share a single connection."

人間が一切介入することなく、AIが `StaticPool` をインポートしてコネクション設定を書き換え、テストをオールグリーンにするまでの軌跡がここに詰まっています。

## 🛠️ 技術スタック

* **AI Agent**: [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) (`/plan` モード)
* **Web Framework**: FastAPI
* **ORM & DB**: SQLAlchemy, SQLite
* **Migration**: Alembic
* **Testing**: pytest, httpx

## 🚀 再現手順（あなたもAIの狂気を体感しよう）

このリポジトリの内容を、自分の手元の空フォルダからClaude Codeに作らせる手順です。
（※Node.js と Python の環境が必要です）

### 1. 準備
```bash
# 空の作業ディレクトリを作成
mkdir claude-code-test && cd claude-code-test

# 仮想環境を作成して有効化
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Claude Codeのインストール（未インストールの場合）
npm install -g @anthropic-ai/claude-code
