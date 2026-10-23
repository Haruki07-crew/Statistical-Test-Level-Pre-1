# 問題の書き方（content/questions/chNN.yaml）

用語辞典の「問題演習」「今日の復習」「この用語の問題」で使う問題の形式と書き方。

## 著作権

- 『統計学実践ワークブック』、公式過去問、Web の記事の**問題文・解説文・数値は使わない**。
- Web の記事は「よく解説される論点」「つまずきやすい点」「典型的な計算の型」を知るためだけに読む。問題・数値・解説はすべて自分で作る。
- 参考にした記事の URL と、そこから拾った論点（1行）を `spec/research/questions/chNN.md` に記録する。

## 目安の問題数

| 用語の重要度 | 問題数 |
|---|---|
| A | 2〜3問 |
| B | 1〜2問 |
| C | 0〜1問 |

1つの用語の問題は、形式と難易度をばらす（例：基本の正誤＋標準の数値＋応用の導出）。

## ファイル形式

```yaml
chapter: 5
questions:
  - id: q05-poisson-distribution-1      # q + 章2桁 + 用語id + 連番。全体で一意
    term: poisson-distribution          # plan.yaml の用語 id（この章の用語）
    level: 標準                          # 基本 / 標準 / 応用
    type: numeric                       # choice / numeric / truefalse / derivation
    question: |
      1時間あたり平均 2.5 件の問い合わせがポアソン分布に従って届く。
      1時間に問い合わせが 1 件以下である確率を小数第3位まで求めよ。
    answer: 0.287                       # 型ごとの書き方は下の表
    tolerance: 0.001                    # numeric のみ必須。絶対誤差
    hint: |
      $P(X\le 1)=P(X=0)+P(X=1)$ を使う。
    explanation: |
      $P(X=0)=e^{-2.5}=0.0821$、$P(X=1)=2.5e^{-2.5}=0.2052$ なので、
      $$P(X\le1)=3.5e^{-2.5}=0.2873$$
```

| type | 必須の項目 | answer の書き方 | 画面での動き |
|---|---|---|---|
| choice（4択） | `choices`（4つ。文字列） | 正解の添字 0〜3 | 選択肢は毎回シャッフル |
| numeric（数値入力） | `tolerance` | 数値 | 入力値と answer の差が tolerance 以内なら正解 |
| truefalse（正誤） | なし | `true` / `false` | 「正しい」「誤り」の2ボタン |
| derivation（導出） | `steps`（2つ以上の文字列） | 最終結果（文字列、数式可） | 「次の式を見る」で途中式を1つずつ表示 → 最終結果 → 自己採点（できた／できなかった） |

- すべての問題に `hint`（答えを言わずに方針を示す）と `explanation`（途中式付きの解説）を書く。
- `question`・`choices`・`hint`・`explanation`・`steps` は Markdown と `$...$`・`$$...$$` の数式が使える。YAML では `|` のブロックで書くと安全（`:` や `#` を含む文字列は必ず引用符で囲む）。
- 数値の問題は「小数第3位まで（四捨五入）」のように答え方を問題文に明記し、tolerance はその丸め幅に合わせる（第3位までなら 0.001 程度、整数なら 0.5 未満）。
- choice の誤答は、よくある間違い（分散と標準偏差の取り違え、自由度の数え違いなど）から作る。
- 流儀が分かれる定義（幾何分布の台、ガンマ分布の尺度/レート）は、問題文で使う定義を明記する。
- 難易度の目安：基本＝定義をそのまま使う／標準＝準1級の典型的な計算1つ／応用＝2段階以上の計算や導出。

## 検算（verify/questions_chNN.py）

数値を含む問題はすべて numpy / scipy / sympy で検算する。検算に通っていない問題はビルドに含まれない。

```python
from qcheck import Q          # verify/qcheck.py
q = Q(5)                      # content/questions/ch05.yaml を読む

from scipy import stats
q.num("q05-poisson-distribution-1", stats.poisson.cdf(1, 2.5))      # numeric: answer と許容誤差で比べる
q.check("q05-poisson-mgf-1", abs(x - 1.5) < 1e-9, "正解の選択肢 1.5")  # 任意の条件（choice/truefalse/derivation）
q.concept("q05-poisson-tf-1", "数値を含まない概念問題")              # 数値を含まない問題だけ
q.done()                      # 集計を表示し、verify/questions_results/ch05.json に保存
```

- `q.num` は numeric 問題の answer を直接比べる。choice・truefalse・derivation で数値を含むものは、`q.check` で「正解の選択肢（または最終結果）の値」と「誤答が正解と一致しないこと」を確かめる。
- 問題の内容（問題文・答え・選択肢など）を書き換えると、検算をやり直すまで公開されない（内容のハッシュで照合する）。
- 実行：`.venv/bin/python verify/questions_ch05.py`（全章は `.venv/bin/python scripts/verify_questions.py`）。

## 検査

`.venv/bin/python scripts/validate.py --questions` で問題ファイルだけを検査できる（通常の validate.py でも検査される）。
