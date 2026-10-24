// iPad 相当（820×1180, WebKit）で主要な画面を開き、描画後に LaTeX の文字が残っていないか確かめる
// 使い方: PW_DIR=<playwright を入れたフォルダ> node verify/math_webkit.mjs [出力フォルダ]
//   docs/ を http://localhost で配信して開く（service worker も動かすため）
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(path.join(process.env.PW_DIR || process.cwd(), 'package.json'));
const { webkit } = require('playwright');
const OUT = process.argv[2] || 'verify/ui_out/webkit';
fs.mkdirSync(OUT, { recursive: true });
const PORT = 8700 + Math.floor(Math.random() * 200);
const srv = spawn('python3', ['-m', 'http.server', String(PORT), '-d', 'docs', '-b', '127.0.0.1'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(800);

// 数式の外に残った LaTeX を探す（mjx-container・入力欄・script の中は除く）
const LEFT = () => {
  const re = /\$[^$\s][^$]*\$|\\(frac|sum|int|mathrm|sqrt|bar|hat|alpha|beta|sigma|mu|theta|lambda|left|right|begin|le|ge|cdot|times)\b/;
  const out = [];
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = w.nextNode())) {
    const p = n.parentElement;
    if (!p || p.closest('mjx-container, script, style, textarea, input, [hidden], .mj-fail')) continue;
    if (p.offsetParent === null && getComputedStyle(p).position !== 'fixed') continue;   // 表示されていない部分
    const m = re.exec(n.textContent);
    if (m) out.push(n.textContent.trim().slice(0, 80));
  }
  return out;
};

async function scrollAll(page) {
  // 解説の枠（スクロールする要素）と画面の両方を下まで送り、遅延描画を起こす
  for (let i = 0; i < 40; i++) {
    const done = await page.evaluate(() => {
      const d = document.getElementById('detail');
      const sc = d && d.scrollHeight > d.clientHeight + 5 ? d : document.scrollingElement;
      const before = sc.scrollTop;
      sc.scrollTop += 900;
      return sc.scrollTop === before;
    });
    await sleep(250);
    if (done) break;
  }
  await page.waitForFunction(() => window.__mathIdle ? window.__mathIdle() : true, null, { timeout: 20000 }).catch(() => {});
  await sleep(1500);
}

const results = [];
async function check(page, name, act) {
  await act();
  await sleep(600);
  await scrollAll(page);
  const left = await page.evaluate(LEFT);
  const math = await page.evaluate(() => document.querySelectorAll('#detail mjx-container').length);
  const errs = await page.evaluate(() => document.querySelectorAll('#detail mjx-merror, #detail [data-mjx-error]').length);
  // 隠れている部分（ヒント・解説）も含めて、組まれていない数式の数
  const raw = await page.evaluate(() => Array.prototype.filter.call(document.querySelectorAll('#detail m, #detail .math-display'), (m) => !m.querySelector('mjx-container')).length);
  const cards = await page.evaluate(() => document.querySelectorAll('#detail .qcard').length);
  results.push({ name, math, errs, raw, cards, left: left.length + raw, sample: left.slice(0, 5) });
  await page.evaluate(() => { const d = document.getElementById('detail'); d.scrollTop = 0; window.scrollTo(0, 0); });
  await page.screenshot({ path: path.join(OUT, name + '.png') });
}

const browser = await webkit.launch();
const ctx = await browser.newContext({ viewport: { width: 820, height: 1180 }, deviceScaleFactor: 1, hasTouch: true, isMobile: false,
  userAgent: 'Mozilla/5.0 (iPad; CPU OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1' });
const page = await ctx.newPage();
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(String(e)));
const base = `http://127.0.0.1:${PORT}/index.html`;
const go = (h) => page.evaluate((h) => { location.hash = h; }, h);
await page.goto(base + '#home');
await page.waitForFunction(() => window.MathJax && window.MathJax.typesetPromise, null, { timeout: 30000 });

const steps = (process.env.STEPS || '').split(',').filter(Boolean);
const plan = [
  ['home', () => go('home')],
  ['term-gamma', () => go('gamma-distribution')],
  ['term-mgf', () => go('moment-generating-function')],
  ['term-likelihood-ratio', () => go('likelihood-ratio-test')],
  ['formulas', () => go('formulas')],
  ['search', async () => { await go('all'); await page.fill('#q', '分散'); await sleep(800); }],
  ['quiz', async () => { await page.fill('#q', ''); await go('home'); await sleep(300); await go('quiz'); await sleep(500); await page.click('[data-start]'); }],
  ['quiz-hint', async () => { await page.click('.qcard [data-act="hint"]'); }],
  ['quiz-answer', async () => {
    for (let i = 0; i < 5; i++) {
      const o = await page.$('.qcard .q-opt:not([disabled])');
      if (o) await o.click();
      else if (await page.$('.qcard .q-num')) { await page.fill('.qcard .q-num', '0'); await page.click('.qcard [data-act="submit"]'); }
      else { let st; while ((st = await page.$('.qcard [data-act="step"]:not([hidden])'))) await st.click(); const sf = await page.$('.qcard [data-self="0"]'); if (sf) await sf.click(); }
      await sleep(300);
      if (i < 4) { const nx = await page.$('[data-next]'); if (nx) { await nx.click(); await sleep(300); } }
    }
  }],
  ['quiz-result', async () => { const nx = await page.$('[data-next]'); if (nx) await nx.click(); }],
  ['exam', async () => { await go('exam'); }],
  ['roadmap', async () => { await go('roadmap'); }],
  ['review', () => go('review')],
  ['stats', () => go('stats')],
];
for (const [name, act] of plan) {
  if (steps.length && !steps.includes(name)) continue;
  try { await check(page, name, act); } catch (e) { results.push({ name, error: String(e).slice(0, 200) }); }
}
// MathJax を読み込めない場合：TeX の文字ではなく案内が出ること
{
  const c2 = await browser.newContext({ viewport: { width: 820, height: 1180 } });
  await c2.route(/mathjax/, (r) => r.abort());
  const p2 = await c2.newPage();
  await p2.goto(base + '#gamma-distribution');
  await sleep(2500);
  const banner = await p2.evaluate(() => { const b = document.querySelector('.mj-banner'); return b ? b.textContent : ''; });
  const left = await p2.evaluate(LEFT);
  await p2.screenshot({ path: path.join(OUT, 'mathjax-failed.png') });
  results.push({ name: 'mathjax-failed', math: 0, errs: 0, raw: 0, cards: 0, left: left.length + (banner.includes('数式を読み込めませんでした') ? 0 : 1), sample: banner ? left.slice(0, 3) : ['案内が出ていない'] });
  await c2.close();
}
await browser.close();
srv.kill();
let bad = 0;
for (const r of results) {
  if (r.error || r.left || r.errs) bad++;
  console.log(`${r.name.padEnd(22)} 数式 ${String(r.math ?? '-').padStart(4)}  エラー ${r.errs ?? '-'}  カード ${r.cards ?? '-'}  未変換 ${r.raw ?? '-'}  残った LaTeX ${r.left ?? '-'}${r.error ? '  失敗: ' + r.error : ''}`);
  for (const s of r.sample || []) console.log('    ' + s);
}
if (pageErrors.length) console.log('ページのエラー:\n  ' + pageErrors.join('\n  '));
console.log(bad ? `NG ${bad} 画面` : 'OK すべての画面で LaTeX の文字は残っていない');
process.exit(bad || pageErrors.length ? 1 : 0);
