// 統計学の教科書（docs/tb/）の動作確認（WebKit）。ホーム・一覧（学年のタブと章の「ねらいと前提」）・用語（教科書で書いた用語と、
// 2級・準1級と共通の用語）・問題を開き、スクリプトのエラー・数式のエラー・はみ出しがないこと、記録のキーが jdt- であること、ほかの級との行き来を確かめる。
// 使い方: PW_DIR=<playwright のフォルダ> node verify/tb_smoke.mjs [出力フォルダ]
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(path.join(process.env.PW_DIR || process.cwd(), 'package.json'));
const { webkit } = require('playwright');
const OUT = process.argv[2] || 'verify/ui_out/tb';
fs.mkdirSync(OUT, { recursive: true });
const PORT = 8800 + Math.floor(Math.random() * 90);
const srv = spawn('python3', ['-m', 'http.server', String(PORT), '-d', 'docs', '-b', '127.0.0.1'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(800);
const base = `http://127.0.0.1:${PORT}`;
const data = JSON.parse(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/.exec(fs.readFileSync('docs/tb/index.html', 'utf8'))[1].replace(/<\\\//g, '</'));
const terms = Object.values(data.terms);
const own = terms.filter((t) => !t.shared), fromG2 = terms.find((t) => t.shared === 'g2'), fromPre1 = terms.find((t) => t.shared === 'pre1');
let ng = 0;
const ok = (c, msg) => { console.log((c ? '[OK] ' : '[NG] ') + msg); if (!c) ng++; };
// 数式のエラー・残った TeX・横はみ出しを調べる
async function mathOk(page) {
  await page.waitForFunction(() => window.MathJax && MathJax.startup && MathJax.startup.promise, null, { timeout: 15000 }).catch(() => {});
  await page.evaluate(() => (window.MathJax && MathJax.startup ? MathJax.startup.promise : null)); await sleep(400);
  return page.evaluate(() => {
    const art = document.querySelector('.article');
    const err = art.querySelectorAll('mjx-merror, [data-mjx-error]').length;
    const raw = [...art.querySelectorAll('.body p, .body li, .body td')].filter((e) => /\$[^$]+\$|\\\(|\\frac|\\sum/.test(e.textContent)).length;
    const over = document.documentElement.scrollWidth > window.innerWidth + 1;
    return { err, raw, over };
  });
}
const browser = await webkit.launch();
for (const w of [375, 1280]) {
  const ctx = await browser.newContext({ viewport: { width: w, height: 900 } });
  const page = await ctx.newPage();
  const errs = [];
  page.on('pageerror', (e) => errs.push(String(e)));
  await page.goto(base + '/tb/#home');
  await page.waitForSelector('.home-hero');
  ok((await page.textContent('.home-kicker')).includes('学部1年〜修士2年'), `${w}px ホームの見出しが教科書`);
  ok((await page.title()).includes('教科書'), `${w}px タイトルが教科書`);
  ok(!(await page.textContent('.article')).includes('本番に向けて'), `${w}px 模擬試験の案内を出さない`);
  ok(!(await page.isVisible('.modes a[data-mode="exam"]')) && !(await page.isVisible('.modes a[data-mode="roadmap"]')), `${w}px メニューに模試・ロードマップを出さない`);
  const links = await page.$$eval('.lvl-link', (as) => as.map((a) => a.getAttribute('href')));
  ok(links.join(',') === '../,../g2/', `${w}px 準1級・2級へのリンク（${links.join(', ')}）`);
  ok((await page.$$('.step')).length === 6, `${w}px 学習の順番が6段階（学部1年〜修士2年）`);
  await page.screenshot({ path: `${OUT}/home-${w}.png` });
  // 一覧：学年のタブと、章を開くと「ねらいと前提」
  await page.goto(base + '/tb/#all'); await page.waitForSelector('.tab');
  const tabs = await page.$$eval('.tab', (b) => b.map((x) => x.textContent));
  ok(tabs.join(',') === '学部1年,学部2年,学部3年,学部4年,修士1年,修士2年', `${w}px 学年のタブ（${tabs.join(',')}）`);
  await page.click('.tab[data-tab="4"]'); await sleep(200);
  await page.click('.ch-head[data-toggle="45"]'); await sleep(200);
  ok((await page.textContent('#list-ch45')).includes('σ-加法族'), `${w}px 第45章を開くと「ねらいと前提」が出る`);
  await page.screenshot({ path: `${OUT}/list-m1-${w}.png` });
  for (const [t, note] of [[fromG2, '2級 用語辞典'], [fromPre1, '準1級 用語辞典']]) {
    await page.goto(base + '/tb/#' + t.id); await page.waitForSelector('.term-title');
    ok((await page.textContent('.see-note')).includes(note), `${w}px 共通の用語（${t.id}）に「${note}」の注記`);
    const crumb = await page.textContent('.crumb');
    ok(crumb.includes('第' + t.chapter + '章'), `${w}px ${t.id} の章が教科書の章（${crumb.trim()}）`);
  }
  for (const t of (w === 375 ? own : own.slice(0, 3))) {
    await page.goto(base + '/tb/#' + t.id); await page.waitForSelector('.term-title');
    const txt = await page.textContent('.article');
    const m = await mathOk(page);
    ok(!txt.includes('辞典」の解説を使っています') && txt.includes('学習のポイント') && m.err === 0 && m.raw === 0 && !m.over,
      `${w}px 教科書の用語 ${t.id}（注記なし・学習のポイント・数式エラー ${m.err}・残った TeX ${m.raw}・はみ出し ${m.over}）`);
    if (w === 375) await page.screenshot({ path: `${OUT}/term-${t.id}-${w}.png`, fullPage: true });
  }
  await page.goto(base + '/tb/#quiz'); await page.waitForSelector('.screen-h'); await sleep(1500);
  const keys = await page.evaluate(() => Object.keys(localStorage));
  ok(keys.includes('jdt-data') && !keys.includes('jd-data') && !keys.includes('jd2-data'), `${w}px 教科書の記録は jdt-data（キー: ${keys.join(', ')}）`);
  await page.goto(base + '/g2/#home'); await page.waitForSelector('.home-hero');
  const l2 = await page.$$eval('.lvl-link', (as) => as.map((a) => a.getAttribute('href')));
  ok(l2.includes('../tb/'), `${w}px 2級のホームから教科書へのリンク（${l2.join(', ')}）`);
  await page.goto(base + '/#home'); await page.waitForSelector('.home-hero');
  const l1 = await page.$$eval('.lvl-link', (as) => as.map((a) => a.getAttribute('href')));
  ok(l1.join(',') === 'g2/,tb/', `${w}px 準1級のホームから2級・教科書へのリンク（${l1.join(', ')}）`);
  ok(errs.length === 0, `${w}px スクリプトのエラーなし` + (errs.length ? '：' + errs.join(' / ') : ''));
  await ctx.close();
}
await browser.close();
srv.kill();
console.log(ng ? `NG ${ng} 件` : '教科書のページ: OK');
process.exit(ng ? 1 : 0);
