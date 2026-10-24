// 2級のページ（docs/g2/）の動作確認（WebKit）。ホーム・用語（2級で書いた用語と準1級と共通の用語）・問題・模擬試験・ロードマップを開き、
// スクリプトのエラーがないこと、記録のキーが級ごとに分かれていること（2級は jd2-）、準1級のページとの行き来を確かめる。
// 使い方: PW_DIR=<playwright のフォルダ> node verify/g2_smoke.mjs [出力フォルダ]
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(path.join(process.env.PW_DIR || process.cwd(), 'package.json'));
const { webkit } = require('playwright');
const OUT = process.argv[2] || 'verify/ui_out/g2';
fs.mkdirSync(OUT, { recursive: true });
const PORT = 8900 + Math.floor(Math.random() * 90);
const srv = spawn('python3', ['-m', 'http.server', String(PORT), '-d', 'docs', '-b', '127.0.0.1'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(800);
const base = `http://127.0.0.1:${PORT}`;
const data = JSON.parse(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/.exec(fs.readFileSync('docs/g2/index.html', 'utf8'))[1].replace(/<\\\//g, '</'));
const own = Object.values(data.terms).find((t) => !t.shared), shared = Object.values(data.terms).find((t) => t.shared);
let ng = 0;
const ok = (c, msg) => { console.log((c ? '[OK] ' : '[NG] ') + msg); if (!c) ng++; };
const browser = await webkit.launch();
for (const w of [375, 1280]) {
  const ctx = await browser.newContext({ viewport: { width: w, height: 900 } });
  const page = await ctx.newPage();
  const errs = [];
  page.on('pageerror', (e) => errs.push(String(e)));
  await page.goto(base + '/g2/#home');
  await page.waitForSelector('.home-hero');
  ok((await page.textContent('.home-kicker')).includes('統計検定2級'), `${w}px ホームの見出しが2級`);
  ok((await page.title()).includes('2級'), `${w}px タイトルが2級`);
  await page.screenshot({ path: `${OUT}/home-${w}.png` });
  if (own) {
    await page.goto(base + '/g2/#' + own.id); await page.waitForSelector('.term-title');
    ok(!(await page.$('.see-note')) || !(await page.textContent('.article')).includes('準1級の辞典と共通'), `${w}px 2級で書いた用語（${own.id}）に共通の注意書きが出ない`);
    await page.screenshot({ path: `${OUT}/term-own-${w}.png`, fullPage: false });
  }
  await page.goto(base + '/g2/#' + shared.id); await page.waitForSelector('.term-title');
  ok((await page.textContent('.article')).includes('準1級の辞典と共通'), `${w}px 共通の用語（${shared.id}）に注意書きが出る`);
  await page.goto(base + '/g2/#exam'); await page.waitForSelector('.screen-h');
  ok((await page.textContent('.article')).includes('35問・90分'), `${w}px 模擬試験の説明が35問・90分`);
  await page.screenshot({ path: `${OUT}/exam-${w}.png` });
  await page.goto(base + '/g2/#quiz'); await page.waitForSelector('.screen-h');
  await page.goto(base + '/g2/#roadmap'); await sleep(300);
  await page.goto(base + '/g2/#all'); await sleep(300);
  const keys = await page.evaluate(() => Object.keys(localStorage));
  ok(keys.includes('jd2-data') && !keys.includes('jd-data'), `${w}px 2級の記録は jd2-data（キー: ${keys.join(', ')}）`);
  await page.goto(base + '/#home'); await page.waitForSelector('.home-hero');
  ok((await page.textContent('.home-kicker')).includes('統計検定準1級'), `${w}px 準1級のページはそのまま`);
  const link = await page.getAttribute('.lvl-link', 'href');
  ok(link === 'g2/', `${w}px 準1級のホームから2級へのリンク（${link}）`);
  await page.goto(base + '/#normal-distribution'); await page.waitForSelector('.term-title'); await sleep(1500);
  const keys2 = await page.evaluate(() => Object.keys(localStorage));
  ok(keys2.includes('jd-data'), `${w}px 準1級の記録は jd-data（キー: ${keys2.join(', ')}）`);
  ok(errs.length === 0, `${w}px スクリプトのエラーなし` + (errs.length ? '：' + errs.join(' / ') : ''));
  await ctx.close();
}
await browser.close();
srv.kill();
console.log(ng ? `NG ${ng} 件` : '2級のページ: OK');
process.exit(ng ? 1 : 0);
