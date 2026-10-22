# 数値検算レポート

## 第1章（事象と確率）

- スクリプト: `verify/ch01.py`（実行: `.venv/bin/python verify/ch01.py`）
- 対象: bayes-theorem, conditional-independence, conditional-probability, independence-of-events, law-of-total-probability
- 結果: **OK 54 / NG 0**
- 方法: Fraction による厳密計算、サイコロ・コイン・玉の全列挙、sympy（逐次更新式＝一括ベイズ、補事象の独立）、乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記：0.0005〜0.01）
- 本文の修正: なし
- 検算対象外: 定性的な記述（現場での使われ方、多変量正規で偏相関0⇔条件付き独立 など他章で扱う主張）

