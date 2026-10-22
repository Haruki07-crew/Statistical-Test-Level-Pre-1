"""すべてのカスタム図を既定値で描く確認用ページ verify/chart_gallery.html を作る（ブラウザで開いて目視・エラー確認）。"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "src"
ids = []
for f in ("charts.js", "charts_custom.js"):
    ids += re.findall(r"(?:registerCustom|reg)\(\s*'([\w-]+)'", (src / f).read_text(encoding="utf-8"))
extra = {"bar-data": "{labels:['PC1','PC2','PC3','PC4'],values:[2.5,0.9,0.4,0.2],cumulative:true}",
         "line-data": "{x:[1,2,3,4],series:[{label:'a',y:[1,3,2,4]}]}", "acf-plot": "{pacf:1}", "km-curve": "{groups:2}",
         "random-walk": "{lower:-10,upper:10,start:0}", "regression-residual": "{outlier:-6}", "points-2d": "{points:[{label:'国語',x:0.8,y:0.4},{label:'数学',x:0.7,y:-0.6}],arrows:1,circle:1,xlabel:'PC1',ylabel:'PC2'}", "graph-diagram": "{nodes:['X1','X2','X3','X4'],edges:[[0,1],[1,2],[2,3],[0,2]]}"}
js = "\n".join((src / f).read_text(encoding="utf-8") for f in ("dist.js", "charts.js", "charts_custom.js"))
items = ",".join(f"['{i}',{extra.get(i, '{}')}]" for i in ids)
html = f"""<!doctype html><html><head><meta charset=utf-8><style>{(src/'app.css').read_text(encoding='utf-8')}
body{{overflow:auto}} .g{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}} h3{{margin:4px 0;font-size:14px}}</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script></head><body>
<pre id=err style="color:red"></pre><div class=g id=g></div><script>{js}</script><script>
window.onerror=function(m){{document.getElementById('err').textContent+=m+'\\n'}};
var origErr=console.error;console.error=function(e){{document.getElementById('err').textContent+='ERR '+(e&&e.message||e)+'\\n';origErr.apply(console,arguments)}};
[{items}].forEach(function(it){{var d=document.createElement('div');d.innerHTML='<h3>'+it[0]+'</h3>';document.getElementById('g').appendChild(d);
Charts.render(d,{{type:'custom',id:it[0],params:it[1],caption:it[0]}});
var m=d.querySelector('.chart-missing');if(m)document.getElementById('err').textContent+=it[0]+': '+m.textContent+'\\n';}});
document.getElementById('err').textContent+='done '+{len(ids)}+'\\n';
</script></body></html>"""
(ROOT / "verify" / "chart_gallery.html").write_text(html, encoding="utf-8")
print(len(ids), "charts")
