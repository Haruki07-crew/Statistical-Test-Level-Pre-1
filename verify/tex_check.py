"""全用語の本文・全問題・公式まとめの数式を MathJax で組み、TeX のエラー（merror）がないか確かめるページ verify/tex_check.html を作る。

使い方: .venv/bin/python verify/tex_check.py のあと、headless Chrome で開いて <pre id="res"> を読む
（例: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --virtual-time-budget=180000 --dump-dom file://$PWD/verify/tex_check.html | grep -o '<pre id="res">[^<]*'）
"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
data = re.search(r'(<script type="application/json" id="data">.*?</script>)', html, re.S).group(1)
page = """<!doctype html><html><head><meta charset=utf-8>
<script>window.MathJax = { tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']], processEscapes: true }, svg: { fontCache: 'global' }, startup: { typeset: false } };</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script></head><body>
<pre id=res>running</pre><div id=g></div>""" + data + """<script>
var D = JSON.parse(document.getElementById('data').textContent), g = document.getElementById('g'), items = [];
Object.keys(D.terms).forEach(function (id) { items.push(['term ' + id, D.terms[id].html]); });
(D.questions || []).forEach(function (q) {
  var parts = [q.q, q.h, q.e].concat(q.c || []).concat(q.steps || []);
  if (q.type === 'derivation') parts.push(q.a);
  items.push(['question ' + q.id, parts.join('<hr>')]);
});
items.push(['formulas', D.formulas || '']);
items.forEach(function (it) { var d = document.createElement('div'); d.dataset.name = it[0]; d.innerHTML = it[1]; g.appendChild(d); });
window.addEventListener('load', function () {
  MathJax.startup.promise.then(function () { return MathJax.typesetPromise([g]); }).then(function () {
    var bad = [];
    g.querySelectorAll('mjx-merror, [data-mjx-error]').forEach(function (e) {
      var host = e.closest('[data-name]'), msg = e.getAttribute('data-mjx-error') || e.textContent;
      bad.push(host.dataset.name + ': ' + msg);
    });
    document.getElementById('res').textContent = 'items ' + items.length + ' / math ' + g.querySelectorAll('mjx-container').length + ' / errors ' + bad.length + '\\n' + bad.slice(0, 50).join('\\n');
  }, function (e) { document.getElementById('res').textContent = 'failed ' + e; });
});
</script></body></html>"""
(ROOT / "verify" / "tex_check.html").write_text(page, encoding="utf-8")
print("verify/tex_check.html を書きました")
