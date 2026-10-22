// docs/index.html に埋め込まれたデータで、検索（src/app.js と同じ正規化・採点）が期待どおり当たるか確かめる
import fs from 'node:fs';
const html = fs.readFileSync(new URL('../docs/index.html', import.meta.url), 'utf8');
const app = fs.readFileSync(new URL('../src/app.js', import.meta.url), 'utf8');
const data = JSON.parse(html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1].replace(/<\\\//g, '</'));
// app.js から norm と scoreEntry をそのまま取り出して使う
const grab = (name) => { const s = app.indexOf('function ' + name); let d = 0, i = app.indexOf('{', s); for (; i < app.length; i++) { if (app[i] === '{') d++; else if (app[i] === '}' && --d === 0) break; } return app.slice(s, i + 1); };
const norm = new Function(grab('norm') + '; return norm;')();
const scoreEntry = new Function(grab('scoreEntry') + '; return scoreEntry;')();
const plainFromHtml = new Function(grab('plainFromHtml') + '; return plainFromHtml;')();
for (const t of Object.values(data.terms)) t.text = plainFromHtml(t.html);
const T = data.terms, ids = data.chapters.flatMap(c => c.terms.map(t => t));
const INDEX = ids.map(p => { const t = T[p.id]; return { id: p.id, names: [p.title].concat(t ? [t.title, ...t.aliases] : []).map(norm), reading: t ? norm(t.reading) : '', en: t ? norm(t.en) : '', tags: t ? t.tags.map(norm) : [], text: t ? norm(t.text) : '' }; });
const search = q => INDEX.map(e => ({ id: e.id, s: scoreEntry(e, norm(q)) })).filter(r => r.s).sort((a, b) => b.s - a.s);
const cases = (process.argv[2] ? JSON.parse(process.argv[2]) : [['ガウス', 'normal-distribution'], ['がんま', 'gamma-distribution'], ['ｶﾞﾝﾏ', 'gamma-distribution'], ['Gamma', 'gamma-distribution'], ['MGF', 'moment-generating-function'], ['ベイズ', 'bayes-theorem'], ['ぽあそん', 'poisson-distribution']]);
let ng = 0;
for (const [q, want] of cases) {
  const r = search(q).slice(0, 5);
  const ok = r.some(x => x.id === want);
  if (!ok) ng++;
  console.log(`${ok ? 'OK' : 'NG'}  「${q}」→ ${r.map(x => `${x.id}(${x.s})`).join(', ') || '0件'}`);
}
console.log(`検索テスト: OK ${cases.length - ng} / NG ${ng}`);
process.exit(ng ? 1 : 0);
