// 教科書（章ごとの本文）の動作確認（WebKit）。目次と、公開しているすべての章を 375px と 1280px で開き、
// スクリプトのエラー・数式のエラー・残った TeX・横のはみ出し・図の描画・解答の開閉・「読み終えた」・章の問題への移動を確かめる。
// 使い方: PW_DIR=<playwright のフォルダ> node verify/book_smoke.mjs <級: pre1|g2|tb> [章のキー ...]
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(path.join(process.env.PW_DIR || process.cwd(), 'package.json'));
const { webkit } = require('playwright');
const LV = process.argv[2] || 'g2';
const ONLY = process.argv.slice(3);
const rel = LV === 'pre1' ? '' : LV + '/';
const OUT = 'verify/ui_out/book_' + LV;
fs.mkdirSync(OUT, { recursive: true });
const PORT = 8700 + Math.floor(Math.random() * 90);
const srv = spawn('python3', ['-m', 'http.server', String(PORT), '-d', 'docs', '-b', '127.0.0.1'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(800);
const url = `http://127.0.0.1:${PORT}/${rel}`;
const data = JSON.parse(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/.exec(fs.readFileSync(`docs/${rel}index.html`, 'utf8'))[1].replace(/<\\\//g, '</'));
const pub = data.book.filter((c) => c.html && (!ONLY.length || ONLY.includes(c.k)));
let ng = 0, okn = 0;
const ok = (c, msg) => { if (c) okn++; else { console.log('[NG] ' + msg); ng++; } };
async function mathDone(page) {
  await page.waitForFunction(() => window.MathJax && MathJax.startup && MathJax.startup.promise, null, { timeout: 15000 }).catch(() => {});
  await page.evaluate(() => (window.MathJax && MathJax.startup ? MathJax.startup.promise : null));
  await sleep(300);
}
const browser = await webkit.launch();
for (const w of [375, 1280]) {
  const ctx = await browser.newContext({ viewport: { width: w, height: 860 } });
  const page = await ctx.newPage();
  const errs = [];
  page.on('pageerror', (e) => errs.push(String(e)));
  await page.goto(url + '#book');
  await sleep(500);
  const rows = await page.$$eval('.book-list li', (l) => l.length);
  ok(rows === data.book.length, `${w}px 目次の行 ${rows} / ${data.book.length}`);
  await page.screenshot({ path: `${OUT}/toc_${w}.png` });
  for (const c of pub) {
    await page.goto(url + '#book/' + c.k);
    await sleep(300);
    await mathDone(page);
    // 下まで少しずつ送り、図と数式を描かせる
    const H = await page.$eval('#detail', (d) => d.scrollHeight);
    for (let y = 0; y < H; y += 700) { await page.$eval('#detail', (d, yy) => { d.scrollTop = yy; }, y); await sleep(120); }
    await sleep(600);
    await mathDone(page);
    const r = await page.evaluate(() => {
      const art = document.querySelector('.article.book');
      const err = art.querySelectorAll('mjx-merror, [data-mjx-error]').length;
      const raw = [...art.querySelectorAll('.body p, .body li, .body td')].filter((e) => /\$[^$]+\$|\\frac|\\sum|\[\[|\{\{chart/.test(e.textContent)).length;
      const over = document.documentElement.scrollWidth > window.innerWidth + 1;
      const wideEls = [...art.querySelectorAll('.body p, .body li, .bx-h, mjx-container:not([display="true"])')].filter((e) => e.getBoundingClientRect().right > window.innerWidth + 1); const wide = wideEls.length; window.__wide = wideEls.map((e) => e.tagName + ':' + e.textContent.slice(0, 50));
      const charts = art.querySelectorAll('.body .chart, .body canvas').length, lazy = art.querySelectorAll('.chart-lazy').length;
      return { err, raw, over, wide, charts, lazy, secs: document.querySelectorAll('.jump button').length, ans: art.querySelectorAll('details.bx-ans').length };
    });
    const nch = (c.html.match(/data-chart=/g) || []).length;
    ok(r.err === 0, `${w}px ${c.k} 数式のエラー ${r.err}`);
    ok(r.raw === 0, `${w}px ${c.k} 組まれずに残った TeX・記法 ${r.raw}`);
    ok(!r.over && r.wide === 0, `${w}px ${c.k} 横のはみ出し（ページ ${r.over}・要素 ${r.wide}）${await page.evaluate(() => JSON.stringify(window.__wide))}`);
    ok(r.lazy === 0 && (nch === 0 || r.charts > 0), `${w}px ${c.k} 図 ${nch} 個（描画 ${r.charts}・未描画 ${r.lazy}）`);
    ok(r.secs >= 3, `${w}px ${c.k} 節へ移動のボタン ${r.secs}`);
    // 解答を開く
    if (r.ans) {
      await page.$eval('details.bx-ans summary', (s) => { s.scrollIntoView(); s.click(); });
      await sleep(200);
      ok(await page.$eval('details.bx-ans', (d) => d.open), `${w}px ${c.k} 解答が開く`);
    }
    if (w === 375 && c === pub[0]) await page.screenshot({ path: `${OUT}/${c.k}_end_${w}.png` });
    // 図の画像（章を指定したときはすべての図、指定しないときは最初の章の 1 枚目）
    const figs = await page.$$('.book .chart');
    for (let i = 0; i < figs.length && (ONLY.length || (c === pub[0] && i === 0)); i++) {
      await figs[i].scrollIntoViewIfNeeded(); await sleep(300); await figs[i].screenshot({ path: `${OUT}/${c.k}_chart${i + 1}_${w}.png` });
    }
    await page.$eval('#detail', (d) => { d.scrollTop = 0; });
    await page.screenshot({ path: `${OUT}/${c.k}_${w}.png` });
  }
  // 読み終えた・章の問題（最初の章で）
  if (pub.length) {
    const c = pub[0];
    await page.goto(url + '#book/' + c.k); await sleep(400);
    const before = await page.$eval('[data-book-read]', (b) => b.textContent);
    await page.click('[data-book-read]'); await sleep(200);
    const after = await page.$eval('[data-book-read]', (b) => b.textContent);
    ok(before !== after && /読み終えた ✓/.test(after), `${w}px 読み終えたの切り替え（${after}）`);
    await page.goto(url + '#book'); await sleep(300);
    ok(await page.$$eval('.bk-done', (l) => l.length) === 1, `${w}px 目次に読了の印`);
    await page.goto(url + '#home'); await sleep(300);
    ok(await page.$eval('.article', (a) => /教科書で読む/.test(a.textContent)), `${w}px ホームに教科書の入口`);
    await page.goto(url + '#book/' + c.k); await sleep(300);
    if (await page.$('[data-book-quiz]')) {
      await page.click('[data-book-quiz]'); await sleep(400);
      ok((await page.evaluate(() => location.hash)) === '#quiz' && !!(await page.$('.sess-card')), `${w}px 章の問題が始まる`);
    }
    // 本文の用語リンクから辞典へ
    await page.goto(url + '#book/' + c.k); await sleep(300);
    const link = await page.$('.book .body .f-link');
    if (link) {
      const id = await link.getAttribute('data-go');
      await link.click(); await sleep(400);
      ok((await page.evaluate(() => location.hash)) === '#' + id && !!(await page.$('.term-title')), `${w}px 用語リンク ${id}`);
    }
  }
  ok(errs.length === 0, `${w}px スクリプトのエラー: ${errs.join(' / ')}`);
  await ctx.close();
}
await browser.close();
srv.kill();
console.log(`教科書の画面テスト（${LV}、${pub.length} 章）: OK ${okn} / NG ${ng}`);
process.exit(ng ? 1 : 0);
