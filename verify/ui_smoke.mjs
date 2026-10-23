// 画面の動作確認：headless Chrome を DevTools プロトコルで動かし、操作・エラー・スクリーンショットを確かめる
// 使い方: node verify/ui_smoke.mjs [出力フォルダ]
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const OUTDIR = process.argv[2] || 'verify/ui_out';
fs.mkdirSync(OUTDIR, { recursive: true });
const URL0 = 'file://' + path.resolve('docs/index.html');
const port = 9333 + Math.floor(Math.random() * 400);
const prof = fs.mkdtempSync(path.join(process.env.TMPDIR || '/tmp', 'jd-ui-'));
const chrome = spawn(CHROME, ['--headless=new', '--disable-gpu', '--hide-scrollbars', `--remote-debugging-port=${port}`, `--user-data-dir=${prof}`, 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let ws, id = 0;
const pending = new Map();
const errors = [];
async function connect() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      const pg = list.find((t) => t.type === 'page');
      if (pg) { ws = new WebSocket(pg.webSocketDebuggerUrl); break; }
    } catch (e) { /* 起動待ち */ }
    await sleep(200);
  }
  await new Promise((r) => ws.addEventListener('open', r));
  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push(m.params.args.map((a) => a.value || a.description).join(' '));
  });
}
function send(method, params = {}) { return new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); }); }
async function ev(expr) {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.result.exceptionDetails) throw new Error(expr + '\n' + JSON.stringify(r.result.exceptionDetails.exception?.description));
  return r.result.result.value;
}
async function size(w, h) { await send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: w < 600 }); }
async function shot(name) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(path.join(OUTDIR, name + '.png'), Buffer.from(r.result.data, 'base64'));
}
async function open(hash, w, h) {
  await size(w, h);
  await send('Page.navigate', { url: 'about:blank' });   // ハッシュだけの移動だと再読み込みされないため
  await sleep(100);
  await send('Page.navigate', { url: URL0 + (hash || '') });
  await sleep(1800);
}
const results = [];
function check(name, ok, info = '') { results.push(ok); console.log(`${ok ? 'OK' : 'NG'}  ${name}${info ? '  ' + info : ''}`); }
const overflow = () => ev(`(() => {
  const bad = document.documentElement.scrollWidth > window.innerWidth + 1 || [...document.querySelectorAll(".pane")].some(p => p.scrollWidth > p.clientWidth + 1);
  if (!bad) return '';
  const out = [];
  document.querySelectorAll('.pane *').forEach(el => {
    const pane = el.closest('.pane'), pr = pane.getBoundingClientRect(), r = el.getBoundingClientRect();
    if (r.width && r.right > pr.right + 1 && !el.closest('.table-wrap, .math-display, .tabs, .jump, .chart-canvas, mjx-container') && out.length < 4) out.push(el.tagName + '.' + el.className + ' ' + Math.round(r.right - pr.right) + 'px');
  });
  return out.join(' | ') || 'overflow';
})()`);

try {
  await connect();
  await send('Page.enable'); await send('Runtime.enable');
  await open('', 1280, 860);
  await ev('localStorage.clear()');
  // 以前の版の「覚えた」を引き継ぐか
  await ev(`localStorage.setItem('jd-memo', JSON.stringify(['bayes-theorem']))`);
  await open('#bayes-theorem', 1280, 860);
  check('旧データの覚えたを引き継ぐ', await ev(`document.querySelector('.memo-toggle').getAttribute('aria-pressed') === 'true'`));
  await sleep(2500);
  check('数式が描かれる', await ev(`document.querySelectorAll('#detail mjx-container').length > 0`), await ev(`document.querySelectorAll('#detail mjx-container').length + '個'`));
  check('読みを表示しない', !(await ev(`document.querySelector('.term-meta')?.textContent.includes('べいず')`)));
  check('この用語の問題がある', await ev(`document.querySelectorAll('#sec-questions .qcard').length > 0`));
  // ブックマーク（b キー）とメモ
  await ev(`document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'b', bubbles: true}))`);
  check('b でブックマーク', await ev(`JSON.parse(localStorage.getItem('jd-data') || '{}').bm?.['bayes-theorem'] > 0 || document.querySelector('.bm-toggle').getAttribute('aria-pressed') === 'true'`));
  await ev(`(() => { const t = document.querySelector('textarea.note'); t.value = 'テストのメモ'; t.dispatchEvent(new Event('input', {bubbles: true})); })()`);
  await sleep(800);
  check('メモの自動保存', await ev(`JSON.parse(localStorage.getItem('jd-data')).notes['bayes-theorem'].x === 'テストのメモ'`));
  // j で次の用語
  await ev(`document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'j', bubbles: true}))`); await sleep(400);
  check('j で次の用語', await ev(`location.hash !== '#bayes-theorem'`), await ev('location.hash'));
  // 問題に答える（用語ページの最初の問題）
  await open('#normal-distribution', 1280, 860);
  const answered = await ev(`(() => {
    const c = document.querySelector('#sec-questions .qcard'); if (!c) return 'no card';
    const o = c.querySelector('.q-opt'); if (o) { o.click(); return 'opt'; }
    const n = c.querySelector('.q-num'); if (n) { n.value = '0'; c.querySelector('[data-act=submit]').click(); return 'num'; }
    const s = c.querySelector('[data-act=step]'); if (s) { while (!s.hidden) s.click(); c.querySelector('[data-self="0"]').click(); return 'der'; }
    return 'none'; })()`);
  check('問題に答えると結果と解説が出る', await ev(`!document.querySelector('#sec-questions .qcard .q-exp').hidden`), answered);
  check('回答が記録される', await ev(`Object.keys(JSON.parse(localStorage.getItem('jd-data')).ans).length > 0`));
  await shot('term-1280');
  // 問題演習：10問を最後まで
  await open('#quiz', 1280, 860);
  await ev(`document.querySelector('[data-count="5"]').click()`);
  await ev(`document.querySelector('[data-start]').click()`);
  // 1問目に★を付ける
  await ev(`document.querySelector('.sess-card .q-star').click()`);
  check('問題の★を保存', await ev(`Object.keys(JSON.parse(localStorage.getItem('jd-data')).qbm).length === 1`));
  for (let i = 0; i < 5; i++) {
    await ev(`(() => { const c = document.querySelector('.sess-card .qcard');
      const o = c.querySelector('.q-opt'); if (o) { o.click(); return; }
      const n = c.querySelector('.q-num'); if (n) { n.value = '1/3'; c.querySelector('[data-act=submit]').click(); return; }
      const s = c.querySelector('[data-act=step]'); if (s) { while (!s.hidden) s.click(); c.querySelector('[data-self="1"]').click(); } })()`);
    if (i === 0) await shot('quiz-1280');
    await ev(`document.querySelector('[data-next]').click()`);
  }
  check('問題演習の結果画面（キーボードでも操作できる）', await ev(`!!document.querySelector('.score')`), await ev(`document.querySelector('.score')?.textContent`));
  await shot('result-1280');
  // 模擬試験モード（時間を計る）
  await open('#quiz', 1280, 860);
  await ev(`document.querySelector('[data-timed]').click()`);
  await ev(`document.querySelector('[data-start]').click()`); await sleep(1200);
  check('模擬試験モードで残り時間を表示', await ev(`/残り \\d+:\\d\\d/.test(document.querySelector('.sess-timer')?.textContent || '')`), await ev(`document.querySelector('.sess-timer')?.textContent`));
  await ev(`document.querySelector('[data-quit]').click()`);
  await ev(`document.querySelector('[data-timed]')?.click()`);
  // 用語カード
  await open('#quiz', 1280, 860);
  await ev(`document.querySelector('[data-cards]').click()`); await sleep(500);
  for (let i = 0; i < 5; i++) {
    await ev(`document.querySelector('[data-card-flip]').click()`);
    if (i === 0) await shot('cards-1280');
    await ev(`document.querySelector('[data-card-judge="${i % 2}"]').click()`);
  }
  check('用語カードを最後まで', await ev(`document.querySelector('.score')?.textContent.includes('覚えた')`), await ev(`document.querySelector('.score')?.textContent`));
  // 試験日
  await open('#stats', 1280, 860);
  await ev(`(() => { const d = document.querySelector('[data-exam]'); d.value = '2026-12-20'; d.dispatchEvent(new Event('change', {bubbles: true})); })()`);
  await open('#all', 1280, 860);
  check('試験日を保存', await ev(`JSON.parse(localStorage.getItem('jd-data')).exam === '2026-12-20'`));
  await open('#review', 1280, 860);
  check('復習画面', await ev(`!!document.querySelector('.screen-h')`), await ev(`document.querySelector('.score')?.textContent`));
  await open('#stats', 1280, 860);
  check('記録画面', await ev(`document.querySelectorAll('.ch-table tbody tr').length >= 32`));
  await ev(`document.querySelector('[data-font="l"]').click()`);
  check('文字サイズ大', await ev(`getComputedStyle(document.documentElement).fontSize === '18px'`));
  await ev(`document.querySelector('[data-font="m"]').click()`);
  await shot('stats-1280');
  // ファイルから読み込む（統合と上書き）
  const payload = JSON.stringify({ app: 'jun1-dictionary', v: 1, exported: '2026-10-20T00:00:00Z', data: { v: 1, memo: { 'gamma-distribution': 5 }, bm: { 't-distribution': 5 }, notes: { 'f-distribution': { x: '別端末のメモ', t: 9 } }, exam: '2027-01-10' } });
  const imp = (mode) => ev(`(async () => {
    const inp = document.querySelector('[data-import]'); const dt = new DataTransfer();
    dt.items.add(new File([${JSON.stringify(payload)}], 'jd.json', { type: 'application/json' }));
    inp.files = dt.files; inp.dispatchEvent(new Event('change', { bubbles: true }));
    await new Promise(r => setTimeout(r, 400));
    document.querySelector('[data-imp="${mode}"]').click();
    const d = JSON.parse(localStorage.getItem('jd-data'));
    return { memo: Object.keys(d.memo).length, note: d.notes['f-distribution'] && d.notes['f-distribution'].x, bm: !!d.bm['t-distribution'] };
  })()`);
  await open('#stats', 1280, 860);
  const memoBefore = await ev(`Object.keys(JSON.parse(localStorage.getItem('jd-data')).memo).length`);
  const m1 = await imp('merge');
  check('読み込み：統合', m1.memo === memoBefore + 1 && m1.note === '別端末のメモ' && m1.bm, JSON.stringify(m1));
  await open('#stats', 1280, 860);
  const m2 = await imp('replace');
  check('読み込み：上書き', m2.memo === 1 && m2.bm, JSON.stringify(m2));
  // 壊れたデータでも落ちない
  await ev(`localStorage.setItem('jd-data', '{"v":1,"memo":[1,2],"ans":{"x":5},"days":"bad"')`);
  await open('#stats', 1280, 860);
  check('壊れたデータでも表示できる', await ev(`!!document.querySelector('.tiles')`));
  await ev(`localStorage.setItem('jd-data', JSON.stringify({v:1, memo:{'x': 'a'}, ans:{'q':{n:'1'}}, days:{'2026-01-01':{a:'x'}}, recent: 'no'}))`);
  await open('#all', 1280, 860);
  check('型の違うデータでも表示できる', await ev(`!!document.querySelector('.tabs')`));
  // 検索・タブ・開閉
  await open('#ch05', 1280, 860);
  check('#ch05 で第5章が開く', await ev(`document.querySelector('[data-toggle="5"]').getAttribute('aria-expanded') === 'true'`));
  await ev(`(() => { const q = document.getElementById('q'); q.value = 'ガウス'; q.dispatchEvent(new Event('input')); })()`); await sleep(300);
  check('検索結果に章名', await ev(`document.querySelector('.row .row-ch')?.textContent.includes('章')`), await ev(`document.querySelector('.row .row-ch')?.textContent`));
  await ev(`(() => { const q = document.getElementById('q'); q.value = ''; q.dispatchEvent(new Event('input')); })()`); await sleep(300);
  check('検索を消すとタブに戻る', await ev(`!!document.querySelector('.tabs')`));
  await open('#formulas', 1280, 860);
  check('公式まとめ', await ev(`!!document.querySelector('.formulas')`));
  // 幅ごとの崩れ
  for (const [w, h] of [[375, 780], [768, 1000], [1280, 860]]) {
    for (const hash of ['#all', '#gamma-distribution', '#quiz', '#stats', '#review', '#formulas', '#cards']) {
      await open(hash, w, h);
      const ov = await overflow();
      check(`${w}px ${hash} 横にはみ出さない`, !ov, ov);
      if (['#gamma-distribution', '#quiz', '#all', '#stats'].includes(hash)) await shot(`${hash.slice(1)}-${w}`);
    }
  }
  // スマホ：一覧 → 解説 → 戻る でスクロール位置を保つ
  await open('#ch06', 375, 780);
  await ev(`document.getElementById('list').scrollTop = 300`);
  const before = await ev(`document.getElementById('list').scrollTop`);
  await ev(`document.querySelector('#list .row:not(.pending) .row-main').click()`); await sleep(600);
  await ev(`document.querySelector('[data-back]').click()`); await sleep(600);
  const after = await ev(`document.getElementById('list').scrollTop`);
  check('スマホで戻るとスクロール位置を保つ', Math.abs(before - after) < 4, `${before} → ${after}`);
  // 初回表示の速さ（DOMContentLoaded まで）
  await open('#gamma-distribution', 1280, 860);
  const dcl = await ev(`Math.round(performance.getEntriesByType('navigation')[0].domContentLoadedEventEnd)`);
  check('初回表示（DOMContentLoaded）が3秒以内', dcl < 3000, dcl + 'ms');
  // ライトモードのホーム
  await ev(`localStorage.clear(); localStorage.setItem('jd-theme', 'light')`);
  await open('', 1280, 900);
  check('ホームに今日の1問', await ev(`!!document.querySelector('.home-q .qcard')`));
  await shot('home-light-1280');
  await open('#quiz', 375, 780);
  await shot('quiz-light-375');
  check('ページのエラーなし', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  console.log('NG  例外: ' + e.message);
  results.push(false);
} finally {
  const ng = results.filter((x) => !x).length;
  console.log(`画面の確認: OK ${results.length - ng} / NG ${ng}`);
  chrome.kill();
  try { fs.rmSync(prof, { recursive: true, force: true }); } catch (e) { /* 無視 */ }
  process.exit(ng ? 1 : 0);
}
