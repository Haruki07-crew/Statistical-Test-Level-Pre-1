"""docs/index.html の全用語の全グラフを描き、エラーや「準備中」が出ないか確かめるページ verify/all_charts.html を作る。"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
data = re.search(r'(<script type="application/json" id="data">.*?</script>)', html, re.S).group(1)
src = ROOT / "src"
js = "\n".join((src / f).read_text(encoding="utf-8") for f in ("dist.js", "charts.js", "charts_custom.js"))
page = f"""<!doctype html><html><head><meta charset=utf-8><style>{(src/'app.css').read_text(encoding='utf-8')} body{{overflow:auto}}</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script></head><body>
<pre id=res></pre><div id=g style="width:700px"></div>{data}<script>{js}</script><script>
var res=document.getElementById('res'), D=JSON.parse(document.getElementById('data').textContent), n=0, bad=[];
var origErr=console.error; console.error=function(e){{bad.push('console: '+(e&&e.message||e));}};
Object.keys(D.terms).forEach(function(id){{ (D.terms[id].charts||[]).forEach(function(spec,i){{
  var d=document.createElement('div'); document.getElementById('g').appendChild(d); n++;
  try {{ Charts.render(d, spec); }} catch(e) {{ bad.push(id+'['+i+'] '+e.message); }}
  var m=d.querySelector('.chart-missing'); if(m) bad.push(id+'['+i+'] '+m.textContent);
  Charts.destroyAll(); d.innerHTML='';
}}); }});
res.textContent='charts '+n+' / problems '+bad.length+'\\n'+bad.join('\\n');
</script></body></html>"""
(ROOT / "verify" / "all_charts.html").write_text(page, encoding="utf-8")
print("verify/all_charts.html を書きました")
