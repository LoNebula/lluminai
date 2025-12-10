# 🧠 Test-time Compute Experiment: Best-of-N Strategy

推論時計算量（Test-time Compute）の有効性と限界を検証するための、**Best-of-N（Majority Voting / Self-Consistency）** 戦略の最小実装です。

本リポジトリのコードは、単純なサンプリング回数の増加（Scaling Inference Compute）が、論理的誤謬（Hallucination）に対してどのような挙動を示すかを確認するために作成されました。

## 🧪 実験の概要

最近の研究（[OpenAI o1](https://openai.com/o1)や[DeepMindの論文](https://arxiv.org/abs/2408.03314)）により、学習時計算量だけでなく「推論時にどれだけ計算リソースを費やすか」が性能向上の鍵であることが示されています。

このプロジェクトでは、以下の仮説と限界を検証します。

  * **仮説**: 回答を複数生成して多数決をとる（Best-of-N）ことで、ランダムな誤りを排除し精度が向上する。
  * **検証結果（限界）**: モデルが体系的な誤り（Systematic Error）を犯す場合、**「誤った回答」で合意形成（Hallucination Consensus）** してしまい、多数決では正解に辿り着けない。

## 🧩 検証に使用した論理パズル

LLMが苦手とする、以下の年齢計算問題を対象としています。

> 「私が6歳のとき、妹は私の半分の年齢でした。そして私が10歳のとき、弟は私の半分の年齢でした。私が70歳になったとき、妹と弟の年齢の合計は何歳ですか？」

  * **正解**: 132歳 (妹67歳 + 弟65歳)
  * **典型的な誤答**: 133歳 (妹68歳 + 弟65歳 など)

## 🚀 セットアップと実行

### 前提条件

  * Python 3.10+
  * OpenAI API Key

### インストール

1.  リポジトリをクローンします。

    ```bash
    git clone https://github.com/your-username/test-time-compute-experiment.git
    cd test-time-compute-experiment
    ```

2.  依存ライブラリをインストールします。

    ```bash
    pip install openai python-dotenv
    ```

3.  環境変数を設定します。
    `.env` ファイルを作成し、APIキーを設定してください。

    ```env
    OPENAI_API_KEY=sk-your-api-key-here
    ```

### 実行

```bash
python main.py
```

## 📊 出力例と考察

`gpt-4o-mini` を使用した実行結果の例です。

```text
--- Running Best-of-5 Strategy ---
Attempt 1: 妹は68歳、弟は65歳です。合計は133歳です。
Attempt 2: 妹は68歳、弟は65歳です。合計は133歳です。
Attempt 3: 妹と弟の年齢の合計は140歳です。
Attempt 4: 妹と弟の年齢の合計は130歳です。
Attempt 5: 妹は68歳、弟は60歳なので、合計は128歳です。
Selected Answer: 妹は68歳、弟は65歳です。合計は133歳です。 (Confidence: 0.40)
```


## 📁 ファイル構成

  * `main.py`: 実験用メインスクリプト
  * `.env`: APIキー設定ファイル（git対象外）