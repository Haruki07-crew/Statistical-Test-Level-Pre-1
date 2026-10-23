// オフラインで開けるか確かめる：localhost で docs/ を配信し、service worker の登録後に通信を切って再読み込みする
// 使い方: node verify/offline_check.mjs
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const hport = 8700 + Math.floor(Math.random() * 200), dport = 9800 + Math.floor(Math.random() * 100);
const server = spawn('.venv/bin/python', ['-m', 'http.server', String(hport), '--bind', '127.0.0.1', '--directory', 'docs'], { stdio: 'ignore' });
const prof = fs.mkdtempSync(path.join(process.env.TMPDIR || '/tmp', 'jd-off-'));
const chrome = spawn(CHROME, ['--headless=new', '--disable-gpu', `--remote-debugging-port=${dport}`, `--user-data-dir=${prof}`, 'about:blank'], { stdio: 'ignore' });
let ws, id = 0; const pending = new Map();
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result.result.value;
let ok = true;
try {
  await sleep(800);
  for (let i = 0; i < 50 && !ws; i++) {
    try { const l = await (await fetch(`http://127.0.0.1:${dport}/json`)).json(); const p = l.find((t) => t.type === 'page'); if (p) ws = new WebSocket(p.webSocketDebuggerUrl); } catch (e) { await sleep(200); }
  }
  await new Promise((r) => ws.addEventListener('open', r));
  ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    if (m.method === 'Runtime.consoleAPICalled' && (m.params.type === 'warning' || m.params.type === 'error')) console.log('  console:', m.params.args.map((a) => a.value || a.description).join(' ').slice(0, 300));
    if (m.method === 'Runtime.exceptionThrown') console.log('  例外:', (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).slice(0, 300)); });
  await send('Page.enable'); await send('Network.enable'); await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 860, deviceScaleFactor: 1, mobile: false });
  const url = `http://localhost:${hport}/#bayes-theorem`;
  await send('Page.navigate', { url });
  await sleep(5000);
  const reg = await ev(`navigator.serviceWorker.ready.then(r => !!r.active)`);
  console.log(`${reg ? 'OK' : 'NG'}  service worker が有効になった`); ok = ok && reg;
  await sleep(2500);   // CDN の保存を待つ
  console.log('  オンライン時の数式:', await ev(`document.querySelectorAll('mjx-container').length`));
  await send('Network.emulateNetworkConditions', { offline: true, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
  await send('Page.reload', { ignoreCache: false });
  await sleep(4000);
  const shown = await ev(`!!document.querySelector('.term-title') && document.querySelector('.term-title').textContent`);
  console.log(`${shown ? 'OK' : 'NG'}  通信を切っても用語ページが開く  ${shown || ''}`); ok = ok && !!shown;
  console.log('  情報:', await ev(`document.readyState + ' / ' + (window.MathJax && MathJax.startup && typeof MathJax.startup.promise) + ' / mjx=' + document.querySelectorAll('mjx-container').length`));
  await sleep(4000);
  const mj = await ev(`!!document.querySelector('mjx-container')`);
  console.log(`${mj ? 'OK' : 'NG'}  オフラインでも数式が描ける（MathJax を保存済み）`); ok = ok && mj;
} catch (e) { console.log('NG  ' + e.message); ok = false; }
console.log(ok ? 'オフライン確認: OK' : 'オフライン確認: NG');
chrome.kill(); server.kill();
try { fs.rmSync(prof, { recursive: true, force: true }); } catch (e) { /* 無視 */ }
process.exit(ok ? 0 : 1);
