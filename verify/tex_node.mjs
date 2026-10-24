// scripts/check_math.py から呼ばれる。JSON の数式をすべて mathjax-full（ブラウザと同じ 3.2.2）で組み、エラーを JSON で返す
// 使い方: MJ_DIR=<mathjax-full を入れたフォルダ> node verify/tex_node.mjs 数式.json
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(path.join(process.env.MJ_DIR, 'package.json'));
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
for (const p of ['base/BaseConfiguration', 'ams/AmsConfiguration', 'newcommand/NewcommandConfiguration', 'boldsymbol/BoldsymbolConfiguration', 'configmacros/ConfigMacrosConfiguration']) require('mathjax-full/js/input/tex/' + p + '.js');
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
// ブラウザでは最初に読み込む拡張（base・ams・newcommand・boldsymbol・configmacros）だけを使う。自動読み込みに頼る命令はエラーとして見つける
const tex = new TeX({ packages: ['base', 'ams', 'newcommand', 'boldsymbol', 'configmacros'], formatError: (jax, err) => { throw err; } });
const doc = mathjax.document('', { InputJax: tex, OutputJax: new SVG({ fontCache: 'none' }) });
const items = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const bad = [];
for (const it of items) {
  try { doc.convert(it.t, { display: it.d }); } catch (e) { bad.push({ d: it.d, t: it.t, err: String(e.message || e).slice(0, 120) }); }
}
process.stdout.write(JSON.stringify(bad));
process.exit(bad.length ? 1 : 0);
