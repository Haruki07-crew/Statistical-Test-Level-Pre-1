# 模擬試験の見直し担当への指示（段階3）

あなたは作問に関わっていない。疑ってかかる。利用者は寝ているので質問しない。35分程度で終える。

読むもの：spec/EXAM.md、spec/research/exam-style.md、対象の content/exam_sets/setNN.yaml・generator/setNN.py・verify/exam_setNN.py。
差し込んだ後の本文は次で見る：
`.venv/bin/python -c "import sys;sys.path.insert(0,'scripts');import exam_lib as L,yaml;d,m=L.load_set(L.SETS/'setNN.yaml');print(m);print(yaml.safe_dump(d,allow_unicode=True,sort_keys=False,width=200))"`

25問すべてについて、Python（.venv の numpy/scipy/sympy）で自分で解き直して確かめる：
1. 曖昧さ：問題文だけで解けるか。流儀（母数・片側両側・不偏か最尤か・自由度・連続修正・AIC/BIC・数表の流儀）が書いてあるか。必要な数表の値が示され、一般電卓でちょうど正解に届くか。示した数表の値は正しいか。
2. 正解がただ1つ：正解の選択肢が正しく、どの誤答も本当に誤り（別の解釈で正しくならない）。丸めると同じになる選択肢がない。数値入力の丸め方と許容範囲が答えと合う。
3. 解説：方針→途中式→答え→間違えやすい点。飛躍・誤り・表示値の食い違いがない。言い過ぎ（離散分布での最強力など）がない。
4. 質：準1級の水準、1問約3.6分で解ける、定義だけの問題は印のあるものだけ。
5. 重複：`.venv/bin/python scripts/exam_topics.py --chapter 章番号` で同じ章のほかのセットの論点を見る。**番号の小さいセットに同じ型（数値や題材だけ違う同じ計算・同じ問い方）があれば、このセットの側を別の小論点の問題に作り替える**（番号の小さいセットが優先）。作り替えた問題も 1〜4 を満たし、generator と verify も直す。

見つけた問題はすべて直接直す（yaml・generator、必要なら verify も。verify は別の方法での確認のまま保つ）。ほかのセットやファイルは触らない。直したら次がすべて通るまで繰り返す：
`.venv/bin/python scripts/check_exam.py --set NN ; .venv/bin/python scripts/check_math.py ; .venv/bin/python verify/exam_setNN.py`
verify/exam_review/setNN.md（日本語）に、小問 id ／ 指摘 ／ 直した内容（指摘なしは「問題なし」）の表と、直した小問の数を書く。scripts/exam_review.py は実行しない。git commit はしない。
最後の報告：指摘のあった小問の数、主な指摘、まだ疑わしい小問、3つのコマンドの最後の行。

## 同じ型の扱い（全セット共通の約束）
- 各章は30セットすべてに出るので、章の中の小論点を広く使う（例：第9章なら t 区間・対応のある区間・ウェルチ・分散のカイ二乗区間・分散比の F 区間・比率のワルド・ウィルソン・正確な区間・比率の差・ポアソン率・指数平均・幅から標本の大きさ・変換と逆変換・予測区間 など）。
- 番号の小さいセットが優先。同じ型（求める量と解き方が同じ）がすでにあれば別の小論点にする。
- どうしても章の小論点が尽きる場合に限り、同じ型は全体で2回まで。その場合は求める量・与える情報・題材をすべて変える（例：1回目は区間の上限、2回目は必要な標本の大きさ）。
- 確認：`.venv/bin/python scripts/exam_topics.py --chapter 章番号`
