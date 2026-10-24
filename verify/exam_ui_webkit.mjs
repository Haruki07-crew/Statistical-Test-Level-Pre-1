// 模擬試験・ロードマップの画面の動作確認（WebKit）。本番モード（解答・印・移動・再読み込みからの再開・採点）、
// 練習モード（答え合わせ）、結果、ロードマップを 375 / 768 / 820 / 1180 / 1280 px で開き、スクリーンショットを撮る。
// 公開中のセットがなければ、既存の問題から作った確認用のセット（FIXTURE）を差し込んで画面だけ確かめる。
// 使い方: PW_DIR=<playwright のフォルダ> node verify/exam_ui_webkit.mjs [出力フォルダ]
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(path.join(process.env.PW_DIR || process.cwd(), 'package.json'));
const { webkit } = require('playwright');
const OUT = process.argv[2] || 'verify/ui_out/exam';
fs.mkdirSync(OUT, { recursive: true });
const PORT = 8900 + Math.floor(Math.random() * 90);
const srv = spawn('python3', ['-m', 'http.server', String(PORT), '-d', 'docs', '-b', '127.0.0.1'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(800);

let html = fs.readFileSync('docs/index.html', 'utf8');
const m = /(<script type="application\/json" id="data">)([\s\S]*?)(<\/script>)/.exec(html);
const data = JSON.parse(m[2].replace(/<\\\//g, '</'));
let fixture = false;
if (!data.exams.length) {
  fixture = true;
  const qs = data.questions.filter((q) => q.type === 'choice' || q.type === 'numeric').slice(0, 300);
  const items = [], probs = [];
  for (let i = 0; i < 25; i++) {
    const q = qs[i * 11];
    if (i % 2 === 0) probs.push({ id: 'p' + i, t: '確認用の大問 ' + (probs.length + 1), ctx: '<p>確認用の状況設定 $\\bar{X}=\\frac{1}{n}\\sum X_i$</p>' });
    const it = { id: 'x' + i, p: probs.length - 1, ch: 1 + (i % 33), lv: '標準', type: q.type, q: q.q, e: q.e, terms: [q.term] };
    if (q.type === 'choice') { it.c = q.c.concat(['確認用の5つ目 $\\sigma^2$']); it.a = q.a; } else { it.a = q.a; it.tol = q.tol || 0.01; it.fmt = '小数第2位まで（±0.01）'; }
    items.push(it);
  }
  data.exams = [{ no: 1, title: '模擬試験 第1回（確認用）', min: 90, pts: 4, probs, items }];
}
if (!data.roadmap) {
  const ids = data.chapters.flatMap((c) => c.terms.map((t) => t.id));
  data.roadmap = { stages: [1, 2, 3, 4].map((n) => ({ no: n, title: '段階' + n, desc: '確認用' })),
    terms: ids.map((id, i) => ({ id, st: 1 + Math.floor(4 * i / ids.length), sc: 100 - Math.floor(100 * i / ids.length), pre: i ? [ids[i - 1]] : [], min: 20, why: '確認用' })) };
  fixture = true;
}
html = html.slice(0, m.index) + m[1] + JSON.stringify(data).replace(/<\//g, '<\\/') + m[3] + html.slice(m.index + m[0].length);
fs.writeFileSync('docs/_exam_test.html', html);

const LEFT = () => {
  const re = /\$[^$\s][^$]*\$|\\(frac|sum|bar|sigma|mathrm|sqrt)\b/;
  const out = [];
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = w.nextNode())) {
    const p = n.parentElement;
    if (!p || p.closest('mjx-container, script, style, [hidden]')) continue;
    if (p.offsetParent === null) continue;
    if (re.test(n.textContent)) out.push(n.textContent.trim().slice(0, 60));
  }
  return out;
};
const browser = await webkit.launch();
const fails = [];
const errors = [];
async function shot(page, name) {
  await page.waitForFunction(() => window.__mathIdle && window.__mathIdle(), null, { timeout: 20000 }).catch(() => {});
  await sleep(400);
  const left = await page.evaluate(LEFT);
  const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  if (left.length) fails.push(`${name}: LaTeX が残っている ${left.slice(0, 2)}`);
  if (over > 2) fails.push(`${name}: 横にはみ出している（${over}px）`);
  await page.screenshot({ path: path.join(OUT, name + '.png') });
}
for (const W of [375, 768, 820, 1180, 1280]) {
  const ctx = await browser.newContext({ viewport: { width: W, height: W < 800 ? 812 : 1000 }, hasTouch: W < 1200 });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(`${W}px: ${e}`));
  const base = `http://127.0.0.1:${PORT}/_exam_test.html`;
  await page.goto(base + '#exam');
  await page.waitForFunction(() => window.MathJax && window.MathJax.typesetPromise, null, { timeout: 30000 });
  await shot(page, `${W}-list`);
  await page.click('.ex-set a');
  await page.click('[data-ex-start="r"]');
  await sleep(300);
  // 本番モード：1問目に答え、印を付け、3問目へ移動
  const opt = await page.$('.ex-opt');
  if (opt) await opt.click(); else await page.fill('.ex-in', '1.23');
  await page.click('[data-ex-flag]');
  await page.click('.ex-g[data-ex-go="2"]');
  const opt2 = await page.$('.ex-opt');
  if (opt2) await opt2.click(); else await page.fill('.ex-in', '0.5');
  await shot(page, `${W}-run`);
  // 再読み込みしても続きから
  await page.reload();
  await sleep(800);
  const st = await page.evaluate(() => ({ hash: location.hash, cur: document.querySelector('.ex-g[aria-current="true"]')?.textContent, done: document.querySelectorAll('.ex-g.done').length, flag: document.querySelectorAll('.ex-g.flag').length, timer: document.querySelector('.ex-timer')?.textContent }));
  if (st.cur !== '3' || st.done !== 2 || st.flag !== 1) fails.push(`${W}px 再開: ${JSON.stringify(st)}`);
  page.on('dialog', (d) => d.accept());
  await page.click('[data-ex-finish]');
  await sleep(600);
  const res = await page.evaluate(() => ({ hash: location.hash, score: document.querySelector('.ex-score b')?.textContent }));
  if (!/result\/\d+/.test(res.hash)) fails.push(`${W}px 採点: ${JSON.stringify(res)}`);
  await shot(page, `${W}-result`);
  // 練習モード
  await page.goto(base + '#exam/1');
  await page.click('[data-ex-start="p"]');
  const o3 = await page.$('.ex-opt');
  if (o3) await o3.click(); else await page.fill('.ex-in', '1');
  await page.click('[data-ex-check]');
  await sleep(300);
  if (!(await page.$('.ex-exp'))) fails.push(`${W}px 練習：解説が出ない`);
  await shot(page, `${W}-practice`);
  await page.goto(base + '#exam');
  await sleep(500);
  await shot(page, `${W}-list-after`);
  await page.goto(base + '#roadmap');
  await sleep(500);
  await page.click('[data-rm-memo]');
  await shot(page, `${W}-roadmap`);
  // 書き出しに模擬試験の記録が入ること
  const exp = await page.evaluate(() => { const d = JSON.parse(localStorage.getItem('jd-data')); return { hist: d.mock.hist.length, cur: !!d.mock.cur }; });
  if (exp.hist !== 1 || !exp.cur) fails.push(`${W}px 保存: ${JSON.stringify(exp)}`);
  await ctx.close();
}
await browser.close();
srv.kill();
fs.unlinkSync('docs/_exam_test.html');
console.log((fixture ? '（確認用のデータを差し込んで確認）\n' : '') + (fails.concat(errors).join('\n') || 'OK 模擬試験・ロードマップの画面（5つの幅）'));
process.exit(fails.length || errors.length ? 1 : 0);
