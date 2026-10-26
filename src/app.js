/* 画面・検索・ルーティング・問題演習・学習記録 */
(function () {
  'use strict';

  var DATA = JSON.parse(document.getElementById('data').textContent);
  var TERMS = DATA.terms;
  var CHAPTERS = DATA.chapters;
  var QS = DATA.questions || [];
  var FORMULAS = DATA.formulas || '';
  // 級ごとの設定（build.py が DATA.site に入れる。準1級は省略された項目に下の既定値を使う）
  var SITE = DATA.site || {}, NAME = SITE.name || '準1級 用語辞典', EXN = (SITE.exam && SITE.exam.n) || 25;
  var PARTS = SITE.parts || [
    ['確率と分布', 1, 6], ['漸近理論と推測', 7, 13], ['確率過程', 14, 15], ['回帰分析', 16, 19],
    ['実験計画と標本調査', 20, 21], ['多変量解析', 22, 26], ['時系列・分割表・不完全データ', 27, 29],
    ['モデル選択・ベイズ・シミュレーション', 30, 32], ['統計的因果推論', 33, 33]
  ];
  var SEC_LABELS = { summary: '一言', definition: '定義', reading: '読み解き', example: '具体例', field: '現場', chart: 'グラフ', exam: SITE.textbook ? '要点' : '試験', mistake: '間違い', questions: '問題', note: 'メモ' };
  var TYPE_LABELS = { choice: '4択', numeric: '数値', truefalse: '正誤', derivation: '導出' };
  var DAY = 86400000;
  var INTERVALS = [1, 3, 7, 14, 30];   // 正解が続いた回数ごとの、次に出すまでの日数

  var $ = function (id) { return document.getElementById(id); };
  var listEl = $('list'), detailEl = $('detail'), qEl = $('q');

  // ---------- 小さな保存（タブ・開閉など。使えなくても動くように） ----------
  // 記録のキーは級ごとに分ける（準1級 jd-、2級 jd2-）。テーマ（jd-theme）だけは共通
  function K(key) { return key === 'jd-theme' || !SITE.kp ? key : SITE.kp + key.slice(3); }
  function load(key, def) { try { var v = localStorage.getItem(K(key)); return v == null ? def : JSON.parse(v); } catch (e) { return def; } }
  function save(key, val) { try { localStorage.setItem(K(key), JSON.stringify(val)); } catch (e) { /* 保存できなくても続ける */ } }

  // ---------- 学習データ（版番号つき。壊れていても落ちない） ----------
  var DB_KEY = K('jd-data'), DB_VERSION = 1;
  function isObj(x) { return x && typeof x === 'object' && !Array.isArray(x); }
  function num(x, def) { return typeof x === 'number' && isFinite(x) ? x : def; }
  function emptyDB() { return { v: DB_VERSION, memo: {}, bm: {}, notes: {}, recent: [], ans: {}, days: {}, searches: [], font: 'm', exam: '', qbm: {}, mock: { hist: [], cur: null } }; }
  // 型の合わない値は捨てて、使える部分だけ残す
  function sanitize(d) {
    var out = emptyDB();
    if (!isObj(d)) return out;
    if (isObj(d.memo)) Object.keys(d.memo).forEach(function (k) { if (d.memo[k]) out.memo[k] = num(d.memo[k], 1); });
    if (isObj(d.bm)) Object.keys(d.bm).forEach(function (k) { if (d.bm[k]) out.bm[k] = num(d.bm[k], 1); });
    if (isObj(d.qbm)) Object.keys(d.qbm).forEach(function (k) { if (d.qbm[k]) out.qbm[k] = num(d.qbm[k], 1); });
    if (isObj(d.notes)) Object.keys(d.notes).forEach(function (k) {
      var n = d.notes[k];
      if (isObj(n) && typeof n.x === 'string' && n.x) out.notes[k] = { x: n.x.slice(0, 20000), t: num(n.t, 0) };
    });
    if (Array.isArray(d.recent)) out.recent = d.recent.filter(function (x) { return typeof x === 'string'; }).slice(0, 30);
    if (isObj(d.ans)) Object.keys(d.ans).forEach(function (k) {
      var a = d.ans[k];
      if (!isObj(a)) return;
      out.ans[k] = { n: num(a.n, 0), c: num(a.c, 0), s: num(a.s, 0), t: num(a.t, 0), d: num(a.d, 0), r: a.r === 1 ? 1 : 0 };
    });
    if (isObj(d.days)) Object.keys(d.days).forEach(function (k) {
      var x = d.days[k];
      if (/^\d{4}-\d{2}-\d{2}$/.test(k) && isObj(x)) out.days[k] = { a: num(x.a, 0), c: num(x.c, 0), v: num(x.v, 0) };
    });
    if (Array.isArray(d.searches)) out.searches = d.searches.filter(function (x) { return typeof x === 'string' && x; }).slice(0, 8);
    if (d.font === 's' || d.font === 'm' || d.font === 'l') out.font = d.font;
    if (typeof d.exam === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(d.exam)) out.exam = d.exam;
    if (isObj(d.mock)) {
      // 模擬試験の記録：解答は選択肢の番号（整数）か、数値入力の文字列だけを残す
      var cleanA = function (a) {
        var o = {};
        if (isObj(a)) Object.keys(a).forEach(function (k) { var v = a[k]; if ((typeof v === 'number' && v >= 0 && v < 5 && v % 1 === 0) || (typeof v === 'string' && v)) o[k] = typeof v === 'string' ? v.slice(0, 40) : v; });
        return o;
      };
      var flags = function (f) { var o = {}; if (isObj(f)) Object.keys(f).forEach(function (k) { if (f[k]) o[k] = 1; }); return o; };
      if (Array.isArray(d.mock.hist)) out.mock.hist = d.mock.hist.filter(isObj).map(function (h) {
        var ch = {};
        if (isObj(h.ch)) Object.keys(h.ch).forEach(function (k) { var a = h.ch[k]; if (Array.isArray(a) && a.length === 2) ch[k] = [num(a[0], 0), num(a[1], 0)]; });
        return { s: num(h.s, 0), sc: num(h.sc, 0), c: num(h.c, 0), n: num(h.n, 0), t: num(h.t, 0), d: num(h.d, 0), m: h.m === 'p' ? 'p' : 'r', ch: ch, a: cleanA(h.a), tu: h.tu ? 1 : 0, v: typeof h.v === 'string' ? h.v.slice(0, 16) : '' };
      }).filter(function (h) { return h.s > 0 && h.t > 0; }).slice(-500);
      var c = d.mock.cur;
      if (isObj(c) && num(c.s, 0) > 0) out.mock.cur = { s: num(c.s, 0), m: c.m === 'p' ? 'p' : 'r', i: num(c.i, 0), left: num(c.left, 0), el: num(c.el, 0), a: cleanA(c.a), f: flags(c.f), k: flags(c.k), st: num(c.st, 0), v: typeof c.v === 'string' ? c.v.slice(0, 16) : '' };
    }
    return out;
  }
  function loadDB() {
    var raw = null, d = null;
    try { raw = localStorage.getItem(DB_KEY); d = raw ? JSON.parse(raw) : null; } catch (e) { d = null; }
    var db = sanitize(d);
    if (!raw) {
      // 以前の版の「覚えた」チェック（jd-memo）を引き継ぐ
      var old = load('jd-memo', []);
      if (Array.isArray(old)) old.forEach(function (id) { if (typeof id === 'string') db.memo[id] = Date.now(); });
      var last = load('jd-last', null);
      if (typeof last === 'string') db.recent = [last];
    }
    return db;
  }
  var DB = loadDB();
  var saveTimer = null, dirty = false;
  function persist(now) {
    clearTimeout(saveTimer);
    dirty = true;
    var w = function () { dirty = false; try { localStorage.setItem(DB_KEY, JSON.stringify(DB)); } catch (e) { /* 容量不足などでも続ける */ } };
    if (now) w(); else saveTimer = setTimeout(w, 200);
  }
  window.addEventListener('pagehide', function () { if (dirty) persist(true); });

  function todayKey(t) {
    var d = t ? new Date(t) : new Date();
    return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
  }
  function bumpDay(field, n) {
    var k = todayKey();
    var d = DB.days[k] || (DB.days[k] = { a: 0, c: 0, v: 0 });
    d[field] += n || 1;
  }

  // ---------- 索引 ----------
  var PLAN = {};      // id → { id, title, importance, status, chapter, order }
  var ORDER = [];     // 学習順の id 一覧
  CHAPTERS.forEach(function (ch) {
    ch.terms.forEach(function (t, i) {
      PLAN[t.id] = { id: t.id, title: t.title, importance: t.importance, status: t.status, chapter: ch.no, order: i, see: t.see || null };
      ORDER.push(t.id);
    });
  });
  function chapterOf(no) { return CHAPTERS[no - 1]; }
  function partOf(no) { for (var i = 0; i < PARTS.length; i++) if (no >= PARTS[i][1] && no <= PARTS[i][2]) return i; return 0; }
  function pad2(n) { return String(n).padStart(2, '0'); }
  function isWritten(id) { return !!TERMS[id]; }

  var QBY = {}, QTERM = {};
  QS.forEach(function (q) { QBY[q.id] = q; (QTERM[q.term] = QTERM[q.term] || []).push(q); });

  // 検索用の正規化：NFKC → 小文字 → カタカナをひらがな → 空白・中黒・ハイフン類を除去
  function norm(s) {
    s = String(s || '');
    try { s = s.normalize('NFKC'); } catch (e) { /* 古い環境 */ }
    s = s.toLowerCase().replace(/[ァ-ヶ]/g, function (c) { return String.fromCharCode(c.charCodeAt(0) - 0x60); });
    return s.replace(/[\s　・･\-‐‑‒–—―−]/g, '');
  }
  // 本文の html から検索用のプレーンテキストを作る（数式は落とす）
  function plainFromHtml(h) {
    return h.replace(/<m>[\s\S]*?<\/m>/g, ' ').replace(/<div class="math-display">[\s\S]*?<\/div>/g, ' ')
      .replace(/<h2>[\s\S]*?<\/h2>/g, ' ').replace(/<[^>]+>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&')
      .replace(/\s+/g, ' ').trim();
  }
  // 本文の検索用テキストは、初回表示を速くするため最初に検索したときに作る
  var textReady = false;
  function ensureText() {
    if (textReady) return;
    textReady = true;
    INDEX.forEach(function (e) { var t = TERMS[e.id]; if (t) { t.text = plainFromHtml(t.html); e.text = norm(t.text); } });
  }

  var INDEX = ORDER.map(function (id) {
    var p = PLAN[id], t = TERMS[id];
    return {
      id: id,
      names: [p.title].concat(t ? [t.title].concat(t.aliases) : []).map(norm),
      reading: t ? norm(t.reading) : '',
      en: t ? norm(t.en) : '',
      tags: t ? t.tags.map(norm) : [],
      text: ''
    };
  });

  function scoreEntry(e, q) {
    var best = 0, i, n;
    for (i = 0; i < e.names.length; i++) {
      n = e.names[i];
      if (!n) continue;
      if (n === q) return 100;
      if (n.indexOf(q) === 0) best = Math.max(best, 80);
      else if (n.indexOf(q) > 0) best = Math.max(best, 60);
    }
    if (best) return best;
    if ((e.reading && e.reading.indexOf(q) >= 0) || (e.en && e.en.indexOf(q) >= 0)) return 50;
    for (i = 0; i < e.tags.length; i++) if (e.tags[i].indexOf(q) >= 0) return 30;
    if (e.text.indexOf(q) >= 0) return 10;
    return 0;
  }

  // 本文で一致した箇所の前後を、元の文字で切り出す
  var mapCache = {};
  function snippet(id, q) {
    var m = mapCache[id];
    if (!m) {
      var src = TERMS[id].text, ns = '', idx = [];
      for (var i = 0; i < src.length; i++) {
        var piece = norm(src[i]);
        for (var j = 0; j < piece.length; j++) { ns += piece[j]; idx.push(i); }
      }
      m = mapCache[id] = { src: src, ns: ns, idx: idx };
    }
    var at = m.ns.indexOf(q);
    if (at < 0) return null;
    var s = m.idx[at], e = m.idx[at + q.length - 1] + 1;
    var a = Math.max(0, s - 20), b = Math.min(m.src.length, e + 20);
    return { pre: (a > 0 ? '…' : '') + m.src.slice(a, s), hit: m.src.slice(s, e), post: m.src.slice(e, b) + (b < m.src.length ? '…' : '') };
  }

  function nearCandidates(q) {
    var grams = [];
    if (q.length === 1) grams = [q];
    for (var i = 0; i + 1 < q.length; i++) grams.push(q.slice(i, i + 2));
    return INDEX.map(function (e) {
      var hay = e.names.join(' ') + ' ' + e.reading + ' ' + e.en, sc = 0;
      grams.forEach(function (g) { if (hay.indexOf(g) >= 0) sc++; });
      return { id: e.id, sc: sc };
    }).filter(function (r) { return r.sc > 0; })
      .sort(function (a, b) { return b.sc - a.sc || ORDER.indexOf(a.id) - ORDER.indexOf(b.id); })
      .slice(0, 6).map(function (r) { return r.id; });
  }

  // ---------- 状態 ----------
  var state = {
    mode: 'dict',           // dict / quiz / review / stats
    tab: 0,                 // 選んでいる部（PARTS の添字）
    open: {},               // 開いている章 { 章番号: true }
    query: '',
    hideMemo: !!load('jd-hide-memo', false),
    onlyBm: false,
    current: null,
    fromList: false,
    dictHash: '#all',
    session: null,          // 解いている問題のまとまり
    searchFocus: false
  };
  (function () {
    var t = Number(load('jd-tab', 0));
    if (t >= 0 && t < PARTS.length) state.tab = t;
    var o = load('jd-open', []);
    if (Array.isArray(o)) o.forEach(function (n) { if (n >= 1 && n <= CHAPTERS.length) state.open[n] = true; });
  })();
  function saveListState() {
    save('jd-tab', state.tab);
    save('jd-open', Object.keys(state.open).filter(function (k) { return state.open[k]; }).map(Number));
  }

  // ---------- 表示の部品 ----------
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function impMark(imp) {
    var title = { A: '重要度A（必須・頻出）', B: '重要度B（重要）', C: '重要度C（補足）' }[imp] || '';
    return '<span class="imp imp-' + imp + '" title="' + title + '" aria-label="' + title + '">' + imp + '</span>';
  }
  var CHECK_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M7.5 12.5l3 3 6-6.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="tick"/></svg>';
  var CHECK_OFF = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>';
  var STAR_ON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z" fill="currentColor" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>';
  var STAR_OFF = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>';

  function chapterCounts(ch) {
    var w = 0;
    ch.terms.forEach(function (t) { if (isWritten(t.id)) w++; });
    return { written: w, total: ch.terms.length };
  }
  function termBtn(id, cls) {
    var p = PLAN[id];
    if (!p) return '';
    return '<button type="button" class="' + (cls || 'rel-btn') + '" data-go="' + id + '"' + (isWritten(id) ? '' : ' disabled') + '>' + esc(p.title) + '</button>';
  }

  // ---------- 数式の描画（どの画面も showMath を使う） ----------
  // 数式は build.py が <m>$…$</m> と <div class="math-display">$$…$$</div> にしている。
  // ・MathJax の読み込み完了を待ってから組む（待っている間は列に積む）
  // ・typesetPromise は同時に走らせず、1本の Promise の列で順番に実行する（同時実行は自動読み込みの再試行と衝突して組み残しが出る）
  // ・隠れている部分（ヒント・解説など）もその場で組む（SVG 出力は表示されていなくても組める）
  // ・読み込みに失敗したら、TeX の文字の代わりに案内を表示する
  var MATH_SEL = 'm, .math-display';
  var mjState = 'wait', mjChain = Promise.resolve(), mjBusy = 0, mjWait = [];
  function rawMath(el) {
    if (!el || el.nodeType !== 1) return [];
    var list = el.matches(MATH_SEL) ? [el] : Array.prototype.slice.call(el.querySelectorAll(MATH_SEL));
    return list.filter(function (m) { return !m.querySelector('mjx-container') && !m.classList.contains('mj-miss'); });
  }
  function showMath(el) {
    if (!el || !rawMath(el).length) return;
    if (mjState === 'fail') { mathFail(el); return; }
    if (mjState !== 'ready') { if (mjWait.indexOf(el) < 0) mjWait.push(el); return; }
    mjBusy++;
    var tries = 0;
    function run() {
      // まだ画面に付いていない要素（作った直後のカードなど）は、付くまで少し待つ
      if (!el.isConnected) { if (++tries > 20) return; return new Promise(function (r) { setTimeout(r, 50); }).then(run); }
      return window.MathJax.typesetPromise([el]);
    }
    mjChain = mjChain.then(run).catch(function (err) { if (window.console) console.warn(err); }).then(function () { mjBusy--; });
  }
  function showMathAll(els) { Array.prototype.forEach.call(els, showMath); }
  function mathFail(el) {
    rawMath(el).forEach(function (m) {
      m.setAttribute('data-tex', m.textContent);
      m.classList.add('mj-miss');
      m.textContent = '［数式］';
    });
    if (!document.querySelector('.mj-banner')) {
      var b = document.createElement('div');
      b.className = 'mj-banner'; b.setAttribute('role', 'alert');
      b.innerHTML = '<span>数式を読み込めませんでした。再読み込みしてください。</span><button type="button" class="q-btn" onclick="location.reload()">再読み込み</button>';
      document.body.appendChild(b);
    }
  }
  function mathReady() {
    if (mjState === 'ready' || !window.MathJax || !window.MathJax.typesetPromise) return;
    var wasFail = mjState === 'fail';
    mjState = 'ready';
    if (wasFail) {   // 遅れて読み込めた場合は、案内を消して元の数式を組み直す
      Array.prototype.forEach.call(document.querySelectorAll('.mj-miss'), function (m) { m.textContent = m.getAttribute('data-tex') || ''; m.classList.remove('mj-miss'); });
      var bn = document.querySelector('.mj-banner'); if (bn) bn.remove();
      showMath(document.body);
    }
    var els = mjWait; mjWait = [];
    els.forEach(function (el) { if (el.isConnected) showMath(el); });
  }
  function mathFailAll() {
    if (mjState !== 'wait') return;
    mjState = 'fail';
    var els = mjWait; mjWait = [];
    els.forEach(mathFail);
  }
  document.addEventListener('mathjax-ready', mathReady);
  document.addEventListener('mathjax-failed', mathFailAll);
  if (window.__mjFailed) setTimeout(mathFailAll, 0);
  // 合図を取りこぼした場合に備えて、MathJax 自身の準備完了の Promise も見る。25秒たっても準備できなければ失敗とする
  window.addEventListener('load', function () {
    var MJ = window.MathJax;
    if (MJ && MJ.startup && MJ.startup.promise) MJ.startup.promise.then(mathReady, mathFailAll);
  });
  setTimeout(function () { if (mjState === 'wait') mathFailAll(); }, 25000);
  // 安全網：showMath を呼び忘れた部分があっても、画面に足された数式は自動で組む
  if (window.MutationObserver) {
    var moQueue = [], moTimer = 0;
    new MutationObserver(function (recs) {
      recs.forEach(function (r) {
        Array.prototype.forEach.call(r.addedNodes, function (n) {
          if (n.nodeType === 1 && !/^MJX-/.test(n.nodeName) && moQueue.indexOf(n) < 0) moQueue.push(n);
        });
      });
      if (moQueue.length && !moTimer) moTimer = setTimeout(function () {
        moTimer = 0;
        var q = moQueue; moQueue = [];
        q.forEach(function (n) { if (n.isConnected) showMath(n); });
      }, 30);
    }).observe(document.body, { childList: true, subtree: true });
  }
  // 自動確認（verify/math_webkit.mjs）用：組み終わったか
  window.__mathIdle = function () { return mjState !== 'wait' && mjBusy === 0 && !mjWait.length; };

  // ---------- グラフの遅延描画（見える部分だけ） ----------
  var io = window.IntersectionObserver ? new IntersectionObserver(function (ents) {
    ents.forEach(function (en) {
      if (!en.isIntersecting) return;
      io.unobserve(en.target);
      activate(en.target);
    });
  }, { root: null, rootMargin: '600px 0px' }) : null;
  function activate(el) {
    if (!el.__spec || !el.isConnected || !el.classList.contains('chart-lazy')) return;
    try { Charts.render(el, el.__spec); } catch (e) { if (window.console) console.error(e); }
    el.classList.remove('chart-lazy');
  }
  // 画面に付いてから監視を始める。監視が働かない環境（古い Safari など）でも 2 秒後には描く
  function lazy(el) {
    if (!io) { setTimeout(function () { activate(el); }, 0); return; }
    setTimeout(function () { if (el.isConnected) io.observe(el); }, 0);
    setTimeout(function () { if (el.isConnected && el.getBoundingClientRect().top < window.innerHeight + 600) activate(el); }, 2000);
  }
  window.addEventListener('load', function () { if (window.Charts) Charts.redrawAll(); });

  function clearDetail() {
    if (window.Charts) Charts.destroyAll();
    if (io) io.disconnect();
    var MJ = window.MathJax;
    if (MJ && MJ.typesetClear) { try { MJ.typesetClear([detailEl]); } catch (e) { /* 無視 */ } }
  }

  // ---------- 用語一覧 ----------
  function rowHtml(id, opts) {
    var p = PLAN[id], t = TERMS[id], written = !!t;
    var cur = state.current === id ? ' aria-current="true"' : '';
    var title = esc(p.title);
    var sub = written && t.en ? esc(t.en) : '';
    var snip = '';
    if (opts && opts.snip) snip = '<span class="row-snippet">' + esc(opts.snip.pre) + '<mark>' + esc(opts.snip.hit) + '</mark>' + esc(opts.snip.post) + '</span>';
    var chTag = opts && opts.showCh ? '<span class="row-ch">第' + p.chapter + '章 ' + esc(chapterOf(p.chapter).title) + '</span>' : '';
    var memoBtn = written
      ? '<button type="button" class="memo-btn" data-memo="' + id + '" aria-pressed="' + !!DB.memo[id] + '" aria-label="覚えた">' + (DB.memo[id] ? CHECK_SVG : CHECK_OFF) + '</button>'
      : '';
    return '<li class="row' + (written ? '' : ' pending') + '" data-id="' + id + '"' + cur + '>' +
      '<button type="button" class="row-main"' + (written ? '' : ' aria-disabled="true" tabindex="-1"') + '>' +
      '<span class="row-title"><span>' + title + '</span>' + impMark(p.importance) + (DB.bm[id] ? '<span class="row-star" aria-label="ブックマーク">' + STAR_ON + '</span>' : '') +
      (written ? '' : '<span class="pending-tag">準備中</span>') + '</span>' +
      (sub ? '<span class="row-en" lang="en">' + sub + '</span>' : '') + chTag + snip + '</button>' + memoBtn + '</li>';
  }

  function visible(id) { return !(state.hideMemo && DB.memo[id]) && !(state.onlyBm && !DB.bm[id]); }

  function listTopHtml(searching) {
    var h = '<div class="list-top">';
    if (!searching) {
      h += '<div class="tabs" role="tablist" aria-label="部">' + PARTS.map(function (p, i) {
        return '<button type="button" class="tab" role="tab" data-tab="' + i + '" aria-selected="' + (state.tab === i) + '">' + esc(p[0]) + '</button>';
      }).join('') + '</div>';
    }
    h += '<div class="list-tools">';
    if (!searching) h += '<button type="button" class="chip" data-all="open">すべて開く</button><button type="button" class="chip" data-all="close">すべて閉じる</button>';
    h += '<button type="button" class="chip memo-chip" data-hide-memo aria-pressed="' + state.hideMemo + '">覚えた用語を隠す</button>' +
      '<button type="button" class="chip" data-only-bm aria-pressed="' + state.onlyBm + '">★だけ</button>' +
      '<a class="chip chip-link" href="#formulas">公式まとめ</a></div>';
    if (searching) h += '<div class="search-head"><h2 class="list-title">「' + esc(state.query) + '」の検索結果 ' + searching.count + '件</h2><p class="list-sub">全章から' + (state.hideMemo ? '（覚えた用語を隠しています）' : '') + '</p></div>';
    else if (state.searchFocus && DB.searches.length) {
      h += '<div class="recent-q"><span>最近の検索</span>' + DB.searches.map(function (s) {
        return '<button type="button" class="chip" data-recent-q="' + esc(s) + '">' + esc(s) + '</button>';
      }).join('') + '</div>';
    }
    return h + '</div>';
  }

  function chapterHtml(n) {
    var ch = chapterOf(n), c = chapterCounts(ch), open = !!state.open[n];
    var hasCur = state.current && PLAN[state.current] && PLAN[state.current].chapter === n;
    var h = '<div class="ch-sec" id="list-ch' + pad2(n) + '">' +
      '<button type="button" class="ch-head' + (hasCur ? ' has-current' : '') + '" data-toggle="' + n + '" aria-expanded="' + open + '" aria-controls="list-ch' + pad2(n) + '-ul">' +
      '<span class="ch-mark" aria-hidden="true">&#x25B6;&#xFE0E;</span><span class="ch-no">' + pad2(n) + '</span><span>' + esc(ch.title) + '</span>' +
      '<span class="ch-count" title="作成済み／予定">' + c.written + '/' + c.total + '</span></button>';
    if (open) {
      if (ch.lead) h += '<p class="ch-lead">' + esc(ch.lead) + '</p>';
      var ids = ch.terms.map(function (t) { return t.id; }).filter(visible);
      h += ids.length ? '<ul id="list-ch' + pad2(n) + '-ul">' + ids.map(function (id) { return rowHtml(id); }).join('') + '</ul>'
        : '<div class="empty" id="list-ch' + pad2(n) + '-ul">表示する用語がありません。</div>';
    }
    return h + '</div>';
  }

  function renderList() {
    var tabs = listEl.querySelector('.tabs'), tabScroll = tabs ? tabs.scrollLeft : 0;
    var h = '', q = norm(state.query);
    if (q) {
      // 検索中はタブを無視して全章から探す
      ensureText();
      var res = [];
      INDEX.forEach(function (e) {
        if (!visible(e.id)) return;
        var s = scoreEntry(e, q);
        if (s) res.push({ id: e.id, s: s });
      });
      res.sort(function (a, b) { return b.s - a.s || ORDER.indexOf(a.id) - ORDER.indexOf(b.id); });
      h += listTopHtml({ count: res.length });
      if (res.length) {
        h += '<ul>' + res.map(function (r) {
          return rowHtml(r.id, { showCh: true, snip: r.s === 10 ? snippet(r.id, q) : null });
        }).join('') + '</ul>';
      } else {
        var near = nearCandidates(q);
        h += '<div class="empty"><h3>見つかりませんでした</h3>' + (state.hideMemo || state.onlyBm ? '<p>「覚えた用語を隠す」「★だけ」を外すと見つかるかもしれません。</p>' : '') +
          (near.length ? '<p>近い候補：</p></div><ul>' + near.map(function (id) { return rowHtml(id, { showCh: true }); }).join('') + '</ul>' : '</div>');
      }
    } else {
      var p = PARTS[state.tab];
      h += listTopHtml(null);
      for (var n = p[1]; n <= p[2]; n++) h += chapterHtml(n);
    }
    listEl.innerHTML = h;
    tabs = listEl.querySelector('.tabs');
    if (tabs) tabs.scrollLeft = tabScroll;
  }

  // 選んでいるタブが横スクロールの外にあれば見える位置へ
  function revealTab() {
    var tabs = listEl.querySelector('.tabs'), b = tabs && tabs.querySelector('.tab[aria-selected="true"]');
    if (!b) return;
    var l = b.offsetLeft - tabs.offsetLeft, r = l + b.offsetWidth;
    if (l < tabs.scrollLeft + 12 || r > tabs.scrollLeft + tabs.clientWidth - 12) tabs.scrollLeft = l - 12;
  }
  function topHeight() { var t = listEl.querySelector('.list-top'); return t ? t.offsetHeight : 0; }
  function scrollChapterIntoView(n) {
    var sec = $('list-ch' + pad2(n));
    if (!sec) return;
    listEl.scrollTop += sec.getBoundingClientRect().top - listEl.getBoundingClientRect().top - topHeight();
  }
  function selectTab(i) {
    if (state.tab === i) return;
    state.tab = i; saveListState();
    renderList(); revealTab();
    listEl.scrollTop = 0;
  }
  function toggleChapter(n) {
    var sec = $('list-ch' + pad2(n)), before = sec ? sec.getBoundingClientRect().top : 0;
    if (state.open[n]) delete state.open[n]; else state.open[n] = true;
    saveListState();
    renderList();
    // 押した見出しが同じ位置に残るようにする
    sec = $('list-ch' + pad2(n));
    if (sec) listEl.scrollTop += sec.getBoundingClientRect().top - before;
    var btn = sec && sec.querySelector('.ch-head');
    if (btn) btn.focus({ preventScroll: true });
  }
  function setAll(open) {
    var p = PARTS[state.tab];
    for (var n = p[1]; n <= p[2]; n++) { if (open) state.open[n] = true; else delete state.open[n]; }
    saveListState(); renderList();
  }

  listEl.addEventListener('mousedown', function (e) {
    // 最近の検索を押したとき、検索欄のフォーカスが外れて一覧が消えないようにする
    if (e.target.closest('[data-recent-q]')) e.preventDefault();
  });
  listEl.addEventListener('click', function (e) {
    var b = e.target.closest('button');
    if (b && b.dataset.tab) { selectTab(Number(b.dataset.tab)); return; }
    if (b && b.dataset.toggle) { toggleChapter(Number(b.dataset.toggle)); return; }
    if (b && b.dataset.all) { setAll(b.dataset.all === 'open'); return; }
    if (b && b.hasAttribute('data-hide-memo')) { state.hideMemo = !state.hideMemo; save('jd-hide-memo', state.hideMemo); renderList(); return; }
    if (b && b.hasAttribute('data-only-bm')) { state.onlyBm = !state.onlyBm; if (state.onlyBm) setAll(true); else renderList(); return; }
    if (b && b.dataset.recentQ) { qEl.value = b.dataset.recentQ; runSearch(); return; }
    var m = e.target.closest('[data-memo]');
    if (m) { toggleMemo(m.dataset.memo); return; }
    var row = e.target.closest('.row');
    if (!row || row.classList.contains('pending')) return;
    if (state.query) rememberSearch(state.query);
    state.fromList = true;
    go('#' + row.dataset.id);
  });

  function toggleMemo(id) {
    if (DB.memo[id]) delete DB.memo[id]; else DB.memo[id] = Date.now();
    persist();
    renderList();
    var mt = detailEl.querySelector('.memo-toggle');
    if (mt && state.current === id) setMemoToggle(mt, id);
  }
  function setMemoToggle(btn, id) {
    btn.setAttribute('aria-pressed', String(!!DB.memo[id]));
    btn.innerHTML = (DB.memo[id] ? CHECK_SVG : CHECK_OFF) + (DB.memo[id] ? '覚えた' : '覚えたらチェック');
  }
  function toggleBm(id) {
    if (DB.bm[id]) delete DB.bm[id]; else DB.bm[id] = Date.now();
    persist();
    renderList();
    var bt = detailEl.querySelector('.bm-toggle');
    if (bt && state.current === id) setBmToggle(bt, id);
  }
  function setBmToggle(btn, id) {
    btn.setAttribute('aria-pressed', String(!!DB.bm[id]));
    btn.innerHTML = (DB.bm[id] ? STAR_ON : STAR_OFF) + (DB.bm[id] ? 'ブックマーク済み' : 'ブックマーク');
  }

  // ---------- ホーム ----------
  // 初学者向けの学習の順番（部をまたいで、土台から積み上げる順）
  var ROADMAP = SITE.steps || [
    { t: '確率と分布の土台', chs: [1, 2, 3, 4, 5, 6],
      why: 'すべての分野の土台です。条件付き確率、期待値と分散の計算、モーメント母関数、主要な分布（二項・ポアソン・正規・カイ二乗・t・F）の性質を、手を動かして確実にします。',
      keys: ['conditional-probability', 'bayes-theorem', 'expectation-variance-properties', 'moment-generating-function', 'change-of-variables', 'binomial-distribution', 'poisson-distribution', 'normal-distribution', 'chi-squared-distribution', 't-distribution', 'f-distribution'] },
    { t: '推定と検定の骨格', chs: [7, 8, 9, 10, 11],
      why: '中心極限定理から始めて、最尤法・フィッシャー情報量、信頼区間、検定の考え方（p値・検出力・ネイマン・ピアソン）へ進みます。準1級の計算問題の多くがこの型です。',
      keys: ['central-limit-theorem', 'delta-method', 'maximum-likelihood-estimator', 'fisher-information', 'confidence-interval', 'p-value', 'power-function', 'neyman-pearson-lemma', 'likelihood-ratio-test', 'two-sample-t-test'] },
    { t: 'いろいろな検定', chs: [12, 13],
      why: '比率・適合度・独立性のカイ二乗検定と、順位を使うノンパラメトリック検定です。どの場面でどの検定を使うかを整理します。',
      keys: ['proportion-test', 'goodness-of-fit-test', 'independence-test', 'wilcoxon-rank-sum-test', 'kruskal-wallis-test', 'spearman-rank-correlation'] },
    { t: '回帰と実験計画・標本調査', chs: [16, 17, 18, 19, 20, 21],
      why: '重回帰の推定と診断、ロジスティック回帰、分散分析、標本調査です。出力表を読み取る問題が多いので、決定係数・t値・F値の意味を押さえます。',
      keys: ['least-squares-method', 'coefficient-of-determination', 'regression-coefficient-t-test', 'multicollinearity', 'cooks-distance', 'logistic-regression', 'one-way-anova', 'stratified-sampling'] },
    { t: '多変量解析', chs: [22, 23, 24, 25, 26],
      why: '主成分分析・判別分析・クラスター分析・因子分析です。固有値と寄与率、マハラノビス距離など、行列の計算の考え方が共通しています。',
      keys: ['principal-component-analysis', 'contribution-ratio', 'linear-discriminant-analysis', 'mahalanobis-distance', 'hierarchical-clustering', 'k-means-method', 'factor-analysis-model'] },
    { t: '発展分野', chs: [14, 15, 27, 28, 29, 30, 31, 32, 33],
      why: 'マルコフ連鎖・確率過程、時系列、分割表、欠測データ、モデル選択、ベイズ法、シミュレーション、因果推論です。土台ができてから、章ごとに取り組みます。',
      keys: ['markov-chain', 'poisson-process', 'ar-model', 'odds-ratio-interval', 'em-algorithm', 'aic', 'conjugate-prior', 'mcmc', 'bootstrap', 'backdoor-criterion'] }
  ];
  function stepTerms(st) { return ORDER.filter(function (id) { return st.chs.indexOf(PLAN[id].chapter) >= 0; }); }
  function stepStats(st) {
    var ids = stepTerms(st), aIds = ids.filter(function (id) { return PLAN[id].importance === 'A'; });
    var memoA = aIds.filter(function (id) { return DB.memo[id]; }).length, n = 0, c = 0;
    QS.forEach(function (q) { if (PLAN[q.term] && st.chs.indexOf(PLAN[q.term].chapter) >= 0 && DB.ans[q.id]) { n += DB.ans[q.id].n; c += DB.ans[q.id].c; } });
    return { aTotal: aIds.length, aMemo: memoA, n: n, c: c, total: ids.length };
  }
  // 次に読むとよい用語：ロードマップ順に、まだ覚えていない・まだ開いていない重要度Aの用語
  function nextTerms(k) {
    var seen = {}, out = [];
    DB.recent.forEach(function (id) { seen[id] = true; });
    ROADMAP.some(function (st) {
      stepTerms(st).some(function (id) {
        if (PLAN[id].importance === 'A' && isWritten(id) && !DB.memo[id] && !seen[id]) out.push(id);
        return out.length >= k;
      });
      return out.length >= k;
    });
    return out;
  }
  function chsLabel(chs) {
    var r = [], i = 0;
    while (i < chs.length) { var j = i; while (j + 1 < chs.length && chs[j + 1] === chs[j] + 1) j++; r.push(i === j ? chs[i] : chs[i] + '〜' + chs[j]); i = j + 1; }
    return '第' + r.join('・') + '章';
  }
  function startStepQuiz(i) {
    var st = ROADMAP[i], pool = shuffle(QS.filter(function (q) { return PLAN[q.term] && st.chs.indexOf(PLAN[q.term].chapter) >= 0 && PLAN[q.term].importance === 'A'; }));
    if (!pool.length) return;
    setMode('quiz');
    startSession('quiz', pool.slice(0, 5), 'ステップ' + (i + 1) + '：' + st.t);
    if (location.hash !== '#quiz') { ignoreNextRoute = true; location.hash = '#quiz'; }
  }
  function startStepCards(i) {
    var st = ROADMAP[i], pool = shuffle(stepTerms(st).filter(function (id) { return isWritten(id) && PLAN[id].importance === 'A' && !DB.memo[id]; }));
    if (!pool.length) pool = shuffle(stepTerms(st).filter(isWritten));
    state.cards = { list: pool.slice(0, 10), i: 0, known: 0, again: [] };
    location.hash === '#cards' ? renderCards() : go('#cards');
  }
  var ignoreNextRoute = false;

  function renderHome() {
    clearDetail();
    var last = DB.recent.filter(isWritten)[0], due = dueList().length, st = streakDays();
    var today = DB.days[todayKey()] || { a: 0, c: 0, v: 0 };
    var fresh = !DB.recent.length && !Object.keys(DB.ans).length && !Object.keys(DB.memo).length;
    var ex = examDays();
    var h = '<div class="article home">' +
      '<header class="home-hero"><p class="home-kicker">' + esc(SITE.kicker || '統計検定準1級') + (SITE.others ? SITE.others.map(function (o) { return ' <a class="lvl-link" href="' + o.href + '">' + esc(o.label) + 'へ →</a>'; }).join('')
        : SITE.other ? ' <a class="lvl-link" href="' + SITE.other.href + '">' + esc(SITE.other.label) + 'の辞典へ →</a>' : '') + '</p><h1>' + esc(SITE.h1 || '用語辞典と問題集') + '</h1>' +
      '<p class="home-lead">' + (SITE.lead ? esc(SITE.lead).replace('{terms}', ORDER.length).replace('{qs}', QS.length) : '『統計学実践ワークブック』（増補版）の全33章に沿って、' + ORDER.length + ' 語の解説と ' + QS.length + ' 問の問題をまとめています。読む・解く・復習するを、毎日少しずつ。') + '</p></header>';
    // 今日の状況
    h += '<div class="home-stats">' +
      '<a href="#review" class="hs"><b>' + due + '</b><span>今日の復習</span></a>' +
      '<a href="#stats" class="hs"><b>' + today.a + '</b><span>今日の回答</span></a>' +
      '<a href="#stats" class="hs"><b>' + st + '<small>日</small></b><span>連続学習</span></a>' +
      (ex !== null ? '<a href="#stats" class="hs"><b>' + (ex >= 0 ? ex + '<small>日</small>' : '—') + '</b><span>試験まで</span></a>'
        : '<a href="#stats" class="hs hs-set"><b>＋</b><span>試験日を設定</span></a>') +
      '</div>';
    // 今日のメニュー
    var nx = nextTerms(3);
    h += '<section class="home-sec"><h2 class="home-h">今日のメニュー<small>15〜30分の目安</small></h2><ol class="menu">' +
      '<li><span class="menu-n">1</span><div><b>復習する</b><p>' + (due ? '期限が来た問題が <b>' + due + '</b> 問あります。先に片付けると記憶に残ります。' : '今日の復習はありません。') + '</p>' +
      (due ? '<a class="q-btn primary" href="#review">復習を始める</a>' : '') + '</div></li>' +
      '<li><span class="menu-n">2</span><div><b>用語を読む</b><p>' + (last && !fresh ? '続きから、または学習の順番で次の用語を読みます。' : 'まずは下の「学習の順番」のステップ1から。読んだら「覚えた」にチェックします。') + '</p><div class="related-list">' +
      (last && !fresh ? '<button type="button" class="rel-btn strong" data-go="' + last + '">続きから：' + esc(PLAN[last].title) + '</button>' : '') +
      nx.map(function (id) { return termBtn(id); }).join('') + '</div></div></li>' +
      '<li><span class="menu-n">3</span><div><b>問題を解く</b><p>読んだ範囲の問題を5問。間違えた問題は自動で「復習」に入ります。</p><a class="q-btn" href="#quiz">問題演習へ</a> <a class="q-btn ghost" href="#cards">用語カードで覚える</a></div></li>' +
      '</ol></section>';
    // 模擬試験とロードマップ
    var lastEx = DB.mock.hist[DB.mock.hist.length - 1], rmn = rmNext(1)[0];
    if (!SITE.textbook) h += '<section class="home-sec"><h2 class="home-h">本番に向けて</h2><div class="ex-modes">' +
      '<a class="ex-mode" href="#exam"><b>模擬試験（' + EXN + '問・90分）</b><span>' + (DB.mock.cur ? '第' + DB.mock.cur.s + '回の途中です。続きから解けます。' : lastEx ? '前回：第' + lastEx.s + '回 ' + lastEx.sc + '点' + (lastEx.sc >= 60 ? '（合格ライン突破）' : '') : '本番と同じ形式で実力を測ります。公開中 ' + EXAMS.length + ' セット。') + '</span></a>' +
      '<a class="ex-mode" href="#roadmap"><b>学習ロードマップ</b><span>' + (RMAP ? '出題されやすさで優先度を付けた4段階の学習順。' + (rmn ? '次は「' + esc(PLAN[rmn.id].title) + '」。' : '') : '準備中です。') + '</span></a>' +
      '</div></section>';
    // 学習の順番
    h += '<section class="home-sec"><h2 class="home-h">はじめての人へ：学習の順番</h2>' +
      '<p class="screen-sub">章の番号順でなくても大丈夫です。土台から積み上げる順に、' + ROADMAP.length + 'つのステップに分けました。各ステップはまず<b>重要度A</b>の用語だけを一通り読み、問題を解いてから、2周目で B・C に進むのがおすすめです。</p><ol class="roadmap">';
    ROADMAP.forEach(function (stp, i) {
      var ss = stepStats(stp), pct = ss.aTotal ? Math.round(100 * ss.aMemo / ss.aTotal) : 0;
      h += '<li class="step"><div class="step-head"><span class="step-n">' + (i + 1) + '</span><div><b>' + esc(stp.t) + '</b><small>' + chsLabel(stp.chs) + '・' + ss.total + '語</small></div></div>' +
        '<p>' + esc(stp.why) + '</p>' +
        '<div class="step-prog"><span class="bar wide"><i style="width:' + pct + '%"></i></span><span>重要度Aを覚えた ' + ss.aMemo + '/' + ss.aTotal + (ss.n ? '・正答率 ' + Math.round(100 * ss.c / ss.n) + '%' : '') + '</span></div>' +
        '<p class="step-keys-h">まず読む用語</p><div class="related-list">' + stp.keys.filter(function (id) { return PLAN[id]; }).map(function (id) {
          return '<button type="button" class="rel-btn' + (DB.memo[id] ? ' done' : '') + '" data-go="' + id + '"' + (isWritten(id) ? '' : ' disabled') + '>' + (DB.memo[id] ? '✓ ' : '') + esc(PLAN[id].title) + '</button>';
        }).join('') + '</div>' +
        '<div class="q-row"><button type="button" class="q-btn" data-step-cards="' + i + '">用語カード</button><button type="button" class="q-btn" data-step-quiz="' + i + '">重要度Aの問題を5問</button>' +
        '<button type="button" class="q-btn ghost" data-goch="' + stp.chs[0] + '">一覧で見る</button></div></li>';
    });
    h += '</ol></section>';
    // 勉強のコツ
    h += '<section class="home-sec"><h2 class="home-h">勉強のコツ</h2><ul class="tips">' + (SITE.tips ? SITE.tips.map(function (x) { return '<li>' + x + '</li>'; }).join('') :
      '<li><b>用語ページは上から順に。</b>「一言でいうと」で全体像をつかみ、「定義・数式」「数式の読み解き」で式の意味を確かめ、「具体例」の数値は自分でも電卓で計算します。</li>' +
      '<li><b>読んだらすぐ解く。</b>ページの最後の「この用語の問題」を解き、間違えたら「よくある間違い」を読み直します。</li>' +
      '<li><b>公式は導けるように。</b>「公式まとめ」で一覧し、期待値・分散・母関数は導出問題で手を動かして確かめます。</li>' +
      '<li><b>毎日少しずつ。</b>復習は 1→3→7→14→30 日と間隔が空くので、毎日「今日の復習」を片付けるのがいちばん効きます。</li>' +
      '<li><b>試験が近づいたら。</b>問題演習の「模擬試験モード」（1問3分）で時間を計り、「苦手」の範囲を集中して解きます。</li>' +
      '<li><b>試験の形式・日程・出題範囲</b>は <a href="https://www.toukei-kentei.jp/" target="_blank" rel="noopener">統計検定の公式サイト</a> で最新の情報を確認してください（2027年から出題範囲表が改訂され、「統計的因果推論」が加わります）。</li>') +
      '</ul></section>';
    // 使い方
    h += '<section class="home-sec"><h2 class="home-h">使い方</h2><div class="howto">' +
      '<details><summary>辞書：用語を探す・読む</summary><ul>' +
      '<li>「辞書」の一覧で、上のタブから部を選び、章の見出しを押すと用語が開きます。「すべて開く／閉じる」もあります。</li>' +
      '<li>上の検索欄に、用語名・読み（ひらがな）・英語名・別名を入れると全章から探せます（「ガウス」「がんま」「MGF」など）。検索欄を押すと最近の検索語が出ます。</li>' +
      '<li>用語ページでは「覚えた」チェック、★ブックマーク、自分のメモ（自動保存）が使えます。上の節ボタンで見たい節へ飛べます。</li>' +
      '<li>PC・iPad では、左上のボタンで左の一覧をたたんで解説を広く読めます。</li>' +
      '<li>重要度の印：' + impMark('A') + ' 必須・頻出　' + impMark('B') + ' 重要　' + impMark('C') + ' 補足</li></ul></details>' +
      '<details><summary>問題：解く・復習する</summary><ul>' +
      '<li>「問題」で出題範囲（全体・部・章・苦手・未回答・ブックマーク）と問題数を選びます。形式は4択・数値入力・正誤・導出（途中式を1つずつ見て自己採点）。</li>' +
      '<li>数値入力は小数・分数（1/3）・%で答えられます。問題文の丸め方（四捨五入の桁）に合わせて入力します。</li>' +
      '<li>答えると「復習」の予定に入ります。間違えた問題はすぐ、正解した問題は 1→3→7→14→30 日後に出ます。</li>' +
      '<li>「模擬試験モード」をオンにすると、1問3分の制限時間がつきます。問題の★で、あとで解き直したい問題を集められます。</li></ul></details>' +
      '<details><summary>用語カード・公式まとめ</summary><ul>' +
      '<li>用語カードは用語名を見て意味を思い出し、めくって「一言でいうと」を確かめます。「覚えた」を押すと覚えたチェックが付きます。</li>' +
      '<li>「公式まとめ」には各分布の期待値・分散・母関数の表と、分野ごとの公式があります。公式から用語ページへ移れます。</li></ul></details>' +
      '<details><summary>記録・設定・データの移し方</summary><ul>' +
      '<li>「記録」で、章ごとの覚えた率と正答率、直近14日の回答数、苦手な用語、最近間違えた問題を見られます。</li>' +
      '<li>文字サイズ（小・中・大）と試験日は「記録」の下で設定します。右上のボタンでライト／ダークを切り替えます。</li>' +
      '<li>記録はこの端末のブラウザに保存されます。iPad とスマホの間で移すときは「記録」の「ファイルに書き出す」→別の端末で「ファイルから読み込む」（統合か上書きを選べます）。</li></ul></details>' +
      '<details><summary>ホーム画面に追加・オフライン</summary><ul>' +
      '<li>iPhone・iPad：Safari で開き、共有ボタン →「ホーム画面に追加」。アプリのように全画面で開けます。</li>' +
      '<li>Android：Chrome のメニュー →「ホーム画面に追加」。</li>' +
      '<li>一度ネットにつないで開くと、次からはオフラインでも開けます（公開ページ https://haruki07-crew.github.io/Statistical-Test-Level-Pre-1/ から開いた場合）。</li></ul></details>' +
      '<details><summary>キーボード操作（PC）</summary><ul>' +
      '<li><kbd>/</kbd> 検索、<kbd>Esc</kbd> 検索を消す、<kbd>j</kbd>／<kbd>k</kbd> 次・前の用語、<kbd>b</kbd> ブックマーク、<kbd>\\</kbd> 左の一覧をたたむ。</li>' +
      '<li>問題演習中：<kbd>1</kbd>〜<kbd>4</kbd> で選択、<kbd>Enter</kbd> で次へ、<kbd>h</kbd> でヒント。</li></ul></details>' +
      '</div></section>';
    // 今日の1問・最近・ブックマーク
    var dq = dueList()[0];
    if (!dq && QS.length) { var seed = Number(todayKey().replace(/-/g, '')); dq = QS[(seed * 2654435761 % 4294967296) % QS.length]; }
    if (dq) h += '<section class="home-sec"><h2 class="home-h">今日の1問</h2><div class="home-q"></div></section>';
    var recent = DB.recent.filter(isWritten).slice(0, 8);
    if (recent.length) h += '<section class="home-sec"><h2 class="home-h">最近見た用語</h2><div class="related-list">' + recent.map(function (id) { return termBtn(id); }).join('') + '</div></section>';
    var bms = Object.keys(DB.bm).filter(isWritten).sort(function (a, b) { return DB.bm[b] - DB.bm[a]; });
    if (bms.length) h += '<section class="home-sec"><h2 class="home-h">ブックマーク</h2><div class="related-list">' + bms.slice(0, 30).map(function (id) { return termBtn(id); }).join('') + '</div></section>';
    h += '<p class="home-note">解説と問題はすべてオリジナルです。数値は Python で検算済み（用語 ' + ORDER.filter(function (id) { return PLAN[id].status === 'verified'; }).length + ' 語・問題 ' + QS.length + ' 問）。<button type="button" class="f-link" data-random>ランダムに1語を開く</button></p></div>';
    detailEl.innerHTML = h;
    var hq = detailEl.querySelector('.home-q');
    if (hq) hq.appendChild(qCard(dq, { termLink: true, retry: true }));
    detailEl.scrollTop = 0;
  }

  // 辞書で用語を選んでいないときの解説欄
  function renderWelcome() {
    clearDetail();
    var last = DB.recent.filter(isWritten)[0];
    detailEl.innerHTML = '<div class="article welcome"><h1>用語を選んでください</h1>' +
      '<p class="screen-sub">左の一覧（スマホでは一覧画面）で部のタブと章を選ぶか、上の検索欄で探します。</p><div class="home-cards">' +
      (last ? '<button type="button" class="home-card" data-go="' + last + '"><small>続きから読む</small><b>' + esc(PLAN[last].title) + '</b></button>' : '') +
      '<a class="home-card" href="#home"><small>ホーム</small><b>使い方・学習の順番</b></a>' +
      '<a class="home-card" href="#formulas"><small>公式まとめ</small><b>分布・検定・推定</b></a>' +
      '<button type="button" class="home-card" data-random><small>ランダムに1語</small><b>どれか開く</b></button></div></div>';
    detailEl.scrollTop = 0;
  }

  function renderPending(id) {
    clearDetail();
    var p = PLAN[id];
    detailEl.innerHTML = '<button type="button" class="back-btn" data-back>← 一覧</button><div class="article"><p class="crumb">第' + p.chapter + '章 ' + esc(chapterOf(p.chapter).title) + '</p>' +
      '<h1 class="term-title">' + esc(p.title) + impMark(p.importance) + '</h1><p class="term-meta"><span class="pending-tag">準備中</span>　この用語の解説はまだ書かれていません。</p></div>';
  }

  function neighbor(id, dir) {
    var p = PLAN[id], terms = chapterOf(p.chapter).terms, i = p.order + dir;
    while (i >= 0 && i < terms.length) {
      if (isWritten(terms[i].id)) return terms[i].id;
      i += dir;
    }
    return null;
  }
  // 章をまたいだ前後（キーボード操作用）
  function globalNeighbor(id, dir) {
    var i = ORDER.indexOf(id) + dir;
    while (i >= 0 && i < ORDER.length) { if (isWritten(ORDER[i])) return ORDER[i]; i += dir; }
    return null;
  }

  // ---------- 用語ページ ----------
  function renderTerm(id) {
    clearDetail();
    var t = TERMS[id], p = PLAN[id], ch = chapterOf(t.chapter), qs = QTERM[id] || [];
    var meta = [];
    if (t.en) meta.push('<span class="en" lang="en">' + esc(t.en) + '</span>');
    if (t.aliases.length) meta.push('別名：' + t.aliases.map(esc).join('、'));
    var h = '<button type="button" class="back-btn" data-back>← 一覧</button><div class="article">' +
      '<p class="crumb">第' + t.chapter + '章 <button type="button" data-goch="' + t.chapter + '">' + esc(ch.title) + '</button></p>' +
      '<h1 class="term-title">' + esc(t.title) + impMark(t.importance) + '</h1>' +
      (meta.length ? '<p class="term-meta">' + meta.join('　・　') + '</p>' : '') +
      '<div class="term-actions"><button type="button" class="memo-toggle"></button><button type="button" class="memo-toggle bm-toggle"></button>' +
      (p.status === 'verified' ? '<span class="pending-tag" style="border-style:solid" title="数値をPythonで検算済み">検算済み</span>' : '') + '</div>';
    if (t.shared && SITE.shared_note) h += '<p class="see-note">' + esc(SITE.shared_note[t.shared] || '') + '</p>';
    else if (t.shared) h += '<p class="see-note">この解説は準1級の辞典と共通で、準1級向けの発展的な内容も含みます。' + (t.note2 ? '<b>2級で押さえること：</b>' + t.note2 : '2級では「一言でいうと」「定義・数式」「具体例」「試験のポイント」を中心に読み、難しい部分は読み飛ばしてかまいません。') + '</p>';
    if (p.see && PLAN[p.see]) h += '<p class="see-note">この内容は「<button type="button" class="f-link" data-go="' + p.see + '">' + esc(PLAN[p.see].title) + '</button>」でまとめて解説しています。このページは補足です。</p>';
    var secs = [];
    t.html.replace(/data-sec="(\w+)"/g, function (_, k) { secs.push(k); return _; });
    if (t.charts.length && secs.indexOf('chart') < 0) secs.splice(Math.max(0, secs.indexOf('exam')), 0, 'chart');
    secs.push('note');
    if (qs.length) secs.push('questions');
    h += '<nav class="jump" aria-label="節へ移動">' + secs.filter(function (k) { return SEC_LABELS[k]; }).map(function (k) {
      return '<button type="button" data-jump="' + k + '">' + SEC_LABELS[k] + (k === 'questions' ? ' ' + qs.length : '') + '</button>';
    }).join('') + '</nav>';
    h += '<div class="body">' + t.html +
      '<section class="sec" data-sec="note" id="sec-note"><h2>自分のメモ</h2><textarea class="note" rows="4" placeholder="覚え方や間違えたところなどを書いておけます（自動で保存）" aria-label="自分のメモ"></textarea><p class="note-status" aria-live="polite"></p></section>' +
      (qs.length ? '<section class="sec" data-sec="questions" id="sec-questions"><h2>この用語の問題</h2><div class="qlist"></div></section>' : '') +
      '</div>';
    if (t.related.length) {
      h += '<section class="related"><h2>関連用語</h2><div class="related-list">' + t.related.map(function (r) {
        var rp = PLAN[r];
        if (!rp) return '';
        return '<button type="button" class="rel-btn" data-go="' + r + '"' + (isWritten(r) ? '' : ' disabled title="準備中"') + '>' + esc(rp.title) + (isWritten(r) ? '' : '（準備中）') + '</button>';
      }).join('') + '</div></section>';
    }
    var prev = neighbor(id, -1), next = neighbor(id, 1);
    h += '<nav class="pager" aria-label="同じ章の前後の用語">' +
      '<button type="button" data-go="' + (prev || '') + '"' + (prev ? '' : ' disabled') + '><small>← 前の用語</small>' + (prev ? esc(PLAN[prev].title) : '') + '</button>' +
      '<button type="button" data-go="' + (next || '') + '"' + (next ? '' : ' disabled') + '><small>次の用語 →</small>' + (next ? esc(PLAN[next].title) : '') + '</button></nav>';
    h += '</div>';
    detailEl.innerHTML = h;
    setMemoToggle(detailEl.querySelector('.memo-toggle:not(.bm-toggle)'), id);
    setBmToggle(detailEl.querySelector('.bm-toggle'), id);

    // メモ（入力が止まったら自動保存）
    var ta = detailEl.querySelector('textarea.note'), ns = detailEl.querySelector('.note-status'), nt = null;
    ta.value = DB.notes[id] ? DB.notes[id].x : '';
    ta.addEventListener('input', function () {
      clearTimeout(nt);
      ns.textContent = '';
      nt = setTimeout(function () {
        var v = ta.value;
        if (v.trim()) DB.notes[id] = { x: v, t: Date.now() }; else delete DB.notes[id];
        persist(true);
        ns.textContent = '保存しました';
      }, 500);
    });

    if (t.charts.length) {
      var sec = detailEl.querySelector('.sec[data-sec="chart"]');
      if (!sec) {
        sec = document.createElement('section');
        sec.className = 'sec'; sec.dataset.sec = 'chart'; sec.id = 'sec-chart';
        sec.innerHTML = '<h2>グラフ・イメージ</h2>';
        var exam = detailEl.querySelector('.sec[data-sec="exam"]');
        exam ? exam.parentNode.insertBefore(sec, exam) : detailEl.querySelector('.body').appendChild(sec);
      }
      t.charts.forEach(function (spec) {
        var holder = document.createElement('div');
        holder.className = 'chart-lazy';
        holder.__spec = spec;
        sec.appendChild(holder);
        lazy(holder);
      });
    }
    if (qs.length) {
      var ql = detailEl.querySelector('.qlist');
      qs.forEach(function (q, i) { ql.appendChild(qCard(q, { label: '問' + (i + 1), retry: true })); });
    }
    detailEl.scrollTop = 0;
    showMathAll(detailEl.querySelectorAll('.term-title, .term-meta, .sec'));   // 上から順に組む
    showMath(detailEl);   // 残り（節の外・問題のカードなど）
    spySetup();
  }

  // 今読んでいる節をジャンプ欄で強調する
  var spyRaf = 0;
  function spyUpdate() {
    spyRaf = 0;
    var jump = detailEl.querySelector('.jump');
    if (!jump) return;
    var top = detailEl.getBoundingClientRect().top + jump.offsetHeight + 24, cur = null;
    Array.prototype.forEach.call(detailEl.querySelectorAll('.body > .sec'), function (s) {
      if (s.getBoundingClientRect().top <= top) cur = s.dataset.sec;
    });
    Array.prototype.forEach.call(jump.querySelectorAll('button'), function (b) {
      var on = b.dataset.jump === cur;
      if (on !== b.classList.contains('on')) {
        b.classList.toggle('on', on);
        if (on) { var l = b.offsetLeft - jump.offsetLeft; if (l < jump.scrollLeft || l + b.offsetWidth > jump.scrollLeft + jump.clientWidth) jump.scrollLeft = l - 16; }
      }
    });
  }
  function spySetup() { spyUpdate(); }
  detailEl.addEventListener('scroll', function () { if (!spyRaf) spyRaf = (window.requestAnimationFrame || setTimeout)(spyUpdate); }, { passive: true });

  // ---------- 問題カード ----------
  function shuffle(a) {
    a = a.slice();
    for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)); var t = a[i]; a[i] = a[j]; a[j] = t; }
    return a;
  }
  // 入力された数値を読む（全角・カンマ・分数 a/b・パーセントに対応）
  function parseNum(s) {
    s = String(s || '');
    try { s = s.normalize('NFKC'); } catch (e) { /* 無視 */ }
    s = s.replace(/[,\s]/g, '').replace(/−/g, '-');
    if (!s) return NaN;
    var pct = /%$/.test(s);
    if (pct) s = s.slice(0, -1);
    var m = /^(-?\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)$/.exec(s);
    var v = m ? Number(m[1]) / Number(m[2]) : (/^-?(\d+\.?\d*|\.\d+)(e-?\d+)?$/i.test(s) ? Number(s) : NaN);
    return pct ? v / 100 : v;
  }
  function fmtNum(x) { return String(Math.round(x * 1e8) / 1e8); }

  function recordAnswer(qid, ok) {
    var now = Date.now(), a = DB.ans[qid] || { n: 0, c: 0, s: 0, t: 0, d: 0, r: 0 };
    a.n++; a.t = now;
    if (ok) { a.c++; a.s++; a.r = 1; a.d = now + INTERVALS[Math.min(a.s, INTERVALS.length) - 1] * DAY; }
    else { a.s = 0; a.r = 0; a.d = now; }
    DB.ans[qid] = a;
    bumpDay('a'); if (ok) bumpDay('c');
    persist(true);
    updateBadge();
  }

  function qCard(q, opts) {
    opts = opts || {};
    var el = document.createElement('article');
    el.className = 'qcard';
    el.dataset.qid = q.id;
    var a = DB.ans[q.id];
    var hist = a ? '<span class="q-hist" title="これまでの正解数／回答数">' + a.c + '/' + a.n + '</span>' : '';
    var head = '<div class="q-meta">' + (opts.label ? '<b>' + esc(opts.label) + '</b>' : '') +
      '<span class="q-tag">' + esc(q.lv) + '</span><span class="q-tag">' + TYPE_LABELS[q.type] + '</span>' + hist +
      '<button type="button" class="q-star" data-act="qbm" aria-pressed="' + !!DB.qbm[q.id] + '" aria-label="この問題をブックマーク" title="この問題をブックマーク">' + (DB.qbm[q.id] ? STAR_ON : STAR_OFF) + '</button>' +
      (opts.termLink ? '<button type="button" class="q-term" data-go="' + q.term + '">' + esc(PLAN[q.term] ? PLAN[q.term].title : q.term) + '</button>' : '') + '</div>';
    var ui = '';
    if (q.type === 'choice') {
      ui = '<div class="q-opts">' + shuffle([0, 1, 2, 3]).map(function (i, k) {
        return '<button type="button" class="q-opt" data-i="' + i + '"><span class="q-key">' + 'ABCD'[k] + '</span><span>' + q.c[i] + '</span></button>';
      }).join('') + '</div>';
    } else if (q.type === 'numeric') {
      ui = '<div class="q-numrow"><input type="text" inputmode="decimal" class="q-num" aria-label="答えの数値" placeholder="数値（分数 1/3 も可）" autocomplete="off"><button type="button" class="q-btn primary" data-act="submit">答える</button></div>';
    } else if (q.type === 'truefalse') {
      ui = '<div class="q-opts tf"><button type="button" class="q-opt" data-tf="1"><span>正しい</span></button><button type="button" class="q-opt" data-tf="0"><span>誤り</span></button></div>';
    } else {
      ui = '<p class="q-note">紙に導出してから、途中式を1つずつ確かめてください。</p><ol class="q-steps"></ol><div class="q-final" hidden><b>結果</b><div>' + q.a + '</div></div>' +
        '<div class="q-row"><button type="button" class="q-btn" data-act="step">途中式を1つ見る（1/' + q.steps.length + '）</button></div>' +
        '<div class="q-self" hidden><span>自己採点：</span><button type="button" class="q-btn primary" data-self="1">できた</button><button type="button" class="q-btn" data-self="0">できなかった</button></div>';
    }
    el.innerHTML = head + '<div class="q-text">' + q.q + '</div>' + ui +
      '<div class="q-row q-tools"><button type="button" class="q-btn ghost" data-act="hint" aria-expanded="false">ヒント</button></div>' +
      '<div class="q-hint" hidden><b>ヒント</b>' + q.h + '</div>' +
      '<div class="q-result" hidden role="status"></div>' +
      '<div class="q-exp" hidden><b>解説</b>' + q.e +
      (opts.termLink && PLAN[q.term] && isWritten(q.term) ? '<p class="q-read"><button type="button" class="rel-btn" data-go="' + q.term + '">「' + esc(PLAN[q.term].title) + '」の解説を読む →</button></p>' : '') + '</div>' +
      '<div class="q-after" hidden></div>';
    var done = false, stepI = 0;
    function finish(ok, detail) {
      if (done) return;
      done = true;
      recordAnswer(q.id, ok);
      var r = el.querySelector('.q-result');
      r.hidden = false;
      r.className = 'q-result ' + (ok ? 'ok' : 'ng');
      r.innerHTML = (ok ? '<b>正解</b>' : '<b>不正解</b>') + (detail ? '<span>' + detail + '</span>' : '');
      el.querySelector('.q-exp').hidden = false;
      el.classList.add('answered');
      Array.prototype.forEach.call(el.querySelectorAll('.q-opt, .q-num, [data-act="submit"]'), function (b) { b.disabled = true; });
      var after = el.querySelector('.q-after');
      if (opts.retry) { after.hidden = false; after.innerHTML = '<button type="button" class="q-btn" data-act="retry">もう一度解く</button>'; }
      showMath(el);
      if (opts.onAnswer) opts.onAnswer(ok, el);
    }
    el.addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (!b || b.disabled) return;
      var act = b.dataset.act;
      if (act === 'hint') {
        var hh = el.querySelector('.q-hint'); hh.hidden = !hh.hidden; b.setAttribute('aria-expanded', String(!hh.hidden));
        if (!hh.hidden) showMath(hh);
        return;
      }
      if (act === 'retry') { el.replaceWith(qCard(q, opts)); return; }
      if (act === 'qbm') {
        if (DB.qbm[q.id]) delete DB.qbm[q.id]; else DB.qbm[q.id] = Date.now();
        persist(true);
        b.setAttribute('aria-pressed', String(!!DB.qbm[q.id])); b.innerHTML = DB.qbm[q.id] ? STAR_ON : STAR_OFF;
        return;
      }
      if (done) return;
      if (b.classList.contains('q-opt') && q.type === 'choice') {
        var i = Number(b.dataset.i), ok = i === q.a;
        b.classList.add(ok ? 'correct' : 'wrong');
        var right = el.querySelector('.q-opt[data-i="' + q.a + '"]');
        if (right) right.classList.add('correct');
        finish(ok, ok ? '' : '正解は「' + right.querySelector('.q-key').textContent + '」');
      } else if (b.dataset.tf) {
        var v = b.dataset.tf === '1', ok2 = v === q.a;
        b.classList.add(ok2 ? 'correct' : 'wrong');
        el.querySelector('.q-opt[data-tf="' + (q.a ? 1 : 0) + '"]').classList.add('correct');
        finish(ok2, ok2 ? '' : '正しくは「' + (q.a ? '正しい' : '誤り') + '」');
      } else if (act === 'submit') {
        var inp = el.querySelector('.q-num'), x = parseNum(inp.value);
        if (isNaN(x)) { inp.focus(); inp.classList.add('bad'); return; }
        var ok3 = Math.abs(x - q.a) <= q.tol + 1e-12;
        finish(ok3, '正解は ' + fmtNum(q.a) + (q.tol ? '（許容誤差 ±' + fmtNum(q.tol) + '）' : '') + '、あなたの答えは ' + fmtNum(x));
      } else if (act === 'step') {
        var ol = el.querySelector('.q-steps'), li = document.createElement('li');
        li.innerHTML = q.steps[stepI++];
        ol.appendChild(li);
        showMath(li);
        if (stepI >= q.steps.length) {
          b.hidden = true;
          var fin = el.querySelector('.q-final'); fin.hidden = false; showMath(fin);
          el.querySelector('.q-self').hidden = false;
        } else b.textContent = '途中式を1つ見る（' + (stepI + 1) + '/' + q.steps.length + '）';
      } else if (b.dataset.self) {
        finish(b.dataset.self === '1', '自己採点');
        el.querySelector('.q-self').hidden = true;
      }
    });
    el.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && e.target.classList.contains('q-num')) { e.preventDefault(); var s = el.querySelector('[data-act="submit"]'); if (s) s.click(); }
    });
    el.addEventListener('input', function (e) { if (e.target.classList.contains('q-num')) e.target.classList.remove('bad'); });
    showMath(el);
    return el;
  }

  // ---------- 復習・苦手 ----------
  function dueList() {
    var now = Date.now();
    return QS.filter(function (q) { var a = DB.ans[q.id]; return a && (a.r === 0 || a.d <= now); })
      .sort(function (x, y) { var a = DB.ans[x.id], b = DB.ans[y.id]; return (a.r - b.r) || (a.d - b.d); });
  }
  function nextDue() {
    var t = Infinity;
    QS.forEach(function (q) { var a = DB.ans[q.id]; if (a && a.r === 1) t = Math.min(t, a.d); });
    return t;
  }
  // 用語ごとの成績（回答のある用語だけ）
  function termStats() {
    var st = {};
    QS.forEach(function (q) {
      var a = DB.ans[q.id];
      if (!a) return;
      var s = st[q.term] || (st[q.term] = { n: 0, c: 0, wrongNow: 0 });
      s.n += a.n; s.c += a.c; if (a.r === 0) s.wrongNow++;
    });
    return st;
  }
  function weakTerms() {
    var st = termStats();
    return Object.keys(st).filter(function (id) { var s = st[id]; return s.wrongNow > 0 || s.c / s.n < 0.6; })
      .sort(function (a, b) { return st[a].c / st[a].n - st[b].c / st[b].n || st[b].wrongNow - st[a].wrongNow; });
  }
  function examDays() {
    if (!DB.exam) return null;
    var p = DB.exam.split('-').map(Number), d = new Date(p[0], p[1] - 1, p[2]), t = new Date();
    t = new Date(t.getFullYear(), t.getMonth(), t.getDate());
    return Math.round((d - t) / DAY);
  }
  function streakDays() {
    var n = 0, d = new Date(), k = todayKey(d.getTime());
    var active = function (key) { var x = DB.days[key]; return x && (x.a > 0 || x.v > 0); };
    if (!active(k)) d = new Date(d.getTime() - DAY);   // 今日まだ何もしていなくても、昨日まで続いていれば数える
    for (;;) {
      k = todayKey(d.getTime());
      if (!active(k)) break;
      n++; d = new Date(d.getTime() - DAY);
    }
    return n;
  }
  function updateBadge() {
    var b = document.querySelector('.modes .badge'), n = dueList().length;
    if (!b) return;
    b.hidden = !n; b.textContent = n > 99 ? '99+' : String(n);
  }

  // ---------- 問題演習 ----------
  var quizCfg = load('jd-quizcfg', null);
  if (!isObj(quizCfg)) quizCfg = {};
  // 保存された設定が壊れていても使えるように整える
  quizCfg = {
    scope: ['all', 'part', 'ch', 'weak', 'new', 'bm'].indexOf(quizCfg.scope) >= 0 ? quizCfg.scope : 'all',
    ch: Number(quizCfg.ch) >= 1 && Number(quizCfg.ch) <= CHAPTERS.length ? Number(quizCfg.ch) : 1,
    part: Number(quizCfg.part) >= 0 && Number(quizCfg.part) < PARTS.length ? Number(quizCfg.part) : 0,
    count: [5, 10, 20].indexOf(Number(quizCfg.count)) >= 0 ? Number(quizCfg.count) : 10,
    timed: quizCfg.timed === true
  };
  var SEC_PER_Q = 180;   // 模擬試験モードの1問あたりの時間（秒）

  function poolFor(cfg) {
    var weak = {};
    weakTerms().forEach(function (id) { weak[id] = true; });
    return QS.filter(function (q) {
      var ch = PLAN[q.term] ? PLAN[q.term].chapter : 0;
      switch (cfg.scope) {
        case 'ch': return ch === Number(cfg.ch);
        case 'part': var p = PARTS[cfg.part] || PARTS[0]; return ch >= p[1] && ch <= p[2];
        case 'weak': return weak[q.term] || (DB.ans[q.id] && DB.ans[q.id].r === 0);
        case 'new': return !DB.ans[q.id];
        case 'bm': return !!DB.bm[q.term] || !!DB.qbm[q.id];
        default: return true;
      }
    });
  }
  var SCOPES = [['all', '全体'], ['part', '部'], ['ch', '章'], ['weak', '苦手'], ['new', '未回答'], ['bm', 'ブックマーク']];

  function renderQuizSetup() {
    clearDetail();
    var pool = poolFor(quizCfg);
    var h = '<div class="article screen"><h1 class="screen-h">問題演習</h1>' +
      '<p class="screen-sub">範囲と問題数を選んで始めます。答えると「今日の復習」の予定に入ります。</p>' +
      '<fieldset class="seg"><legend>出題範囲</legend><div class="seg-row">' + SCOPES.map(function (s) {
        return '<button type="button" class="chip" data-scope="' + s[0] + '" aria-pressed="' + (quizCfg.scope === s[0]) + '">' + s[1] + '</button>';
      }).join('') + '</div>';
    if (quizCfg.scope === 'part') h += '<label class="sel">部：<select data-cfg="part">' + PARTS.map(function (p, i) { return '<option value="' + i + '"' + (Number(quizCfg.part) === i ? ' selected' : '') + '>' + esc(p[0]) + '</option>'; }).join('') + '</select></label>';
    if (quizCfg.scope === 'ch') h += '<label class="sel">章：<select data-cfg="ch">' + CHAPTERS.map(function (c) {
      var n = QS.filter(function (q) { return PLAN[q.term] && PLAN[q.term].chapter === c.no; }).length;
      return '<option value="' + c.no + '"' + (Number(quizCfg.ch) === c.no ? ' selected' : '') + (n ? '' : ' disabled') + '>' + pad2(c.no) + ' ' + esc(c.title) + '（' + n + '問）</option>';
    }).join('') + '</select></label>';
    h += '</fieldset><fieldset class="seg"><legend>問題数</legend><div class="seg-row">' + [5, 10, 20].map(function (n) {
      return '<button type="button" class="chip" data-count="' + n + '" aria-pressed="' + (Number(quizCfg.count) === n) + '">' + n + '問</button>';
    }).join('') + '</div></fieldset>' +
      '<fieldset class="seg"><legend>時間制限（本番形式の模擬試験は「模試」から）</legend><div class="seg-row"><button type="button" class="chip" data-timed aria-pressed="' + quizCfg.timed + '">時間を計る（1問' + SEC_PER_Q / 60 + '分）</button></div></fieldset>' +
      '<p class="screen-sub">この範囲の問題：<b>' + pool.length + '</b> 問' + (pool.length ? '' : '（' + emptyReason(quizCfg.scope) + '）') + '</p>' +
      '<button type="button" class="big-btn" data-start' + (pool.length ? '' : ' disabled') + '>始める</button>' +
      '<button type="button" class="big-btn ghost" data-cards>この範囲の用語カードで覚える（' + cardPool().length + '語）</button>';
    if (state.session && state.session.kind === 'quiz' && state.session.i < state.session.list.length) h += '<button type="button" class="big-btn ghost" data-resume>解いている途中の問題に戻る（' + (state.session.i + 1) + '/' + state.session.list.length + '）</button>';
    h += '</div>';
    detailEl.innerHTML = h;
    detailEl.scrollTop = 0;
  }
  function emptyReason(scope) {
    return { weak: '苦手な用語はまだありません。問題を解くと、間違えた用語がここに入ります', new: 'すべて回答済みです', bm: '★を付けた問題や、★を付けた用語の問題がありません' }[scope] || 'この範囲には問題がまだありません';
  }
  function startSession(kind, list, title) {
    state.session = { kind: kind, list: list.map(function (q) { return q.id; }), i: 0, res: [], title: title, answered: false, start: Date.now() };
    if (kind === 'quiz' && quizCfg.timed) { state.session.deadline = Date.now() + list.length * SEC_PER_Q * 1000; state.session.title = '模擬試験'; }
    renderSession();
  }
  function renderSession() {
    clearDetail();
    var s = state.session;
    if (!s) { state.mode === 'review' ? renderReview() : renderQuizSetup(); return; }
    if (s.i >= s.list.length) { renderResult(); return; }
    var q = QBY[s.list[s.i]];
    s.answered = false;
    detailEl.innerHTML = '<div class="article screen"><div class="sess-head"><button type="button" class="q-btn ghost" data-quit>やめる</button><b>' + esc(s.title) + '</b>' + (s.deadline ? '<span class="sess-timer" aria-live="off">' + fmtLeft(s.deadline - Date.now()) + '</span>' : '') + '<span class="sess-n">' + (s.i + 1) + ' / ' + s.list.length + '</span></div>' +
      '<div class="sess-bar" aria-hidden="true"><span style="width:' + (100 * s.i / s.list.length) + '%"></span></div><div class="sess-card"></div>' +
      '<div class="sess-next" hidden><button type="button" class="big-btn" data-next>' + (s.i + 1 < s.list.length ? '次の問題へ' : '結果を見る') + '</button></div></div>';
    var card = qCard(q, {
      termLink: true,
      onAnswer: function (ok) {
        s.res.push({ id: q.id, ok: ok });
        s.answered = true;
        var nx = detailEl.querySelector('.sess-next'); nx.hidden = false;
      }
    });
    detailEl.querySelector('.sess-card').appendChild(card);
    detailEl.scrollTop = 0;
    card.tabIndex = -1;
    try { card.focus({ preventScroll: true }); } catch (e) { /* 無視 */ }
  }
  function fmtLeft(ms) {
    var t = Math.max(0, Math.round(ms / 1000));
    return '残り ' + Math.floor(t / 60) + ':' + String(t % 60).padStart(2, '0');
  }
  function fmtTime(ms) { var t = Math.round(ms / 1000); return Math.floor(t / 60) + '分' + String(t % 60).padStart(2, '0') + '秒'; }
  // 模擬試験の残り時間（1秒ごと）。時間切れになったら、そこまでの回答で結果を出す
  setInterval(function () {
    var s = state.session;
    if (!s || !s.deadline || s.done) return;
    var left = s.deadline - Date.now(), el = detailEl.querySelector('.sess-timer');
    if (el) { el.textContent = fmtLeft(left); el.classList.toggle('low', left < 60000); }
    if (left <= 0) {
      s.timeUp = s.list.length - s.res.length;
      s.list = s.list.slice(0, s.res.length); s.i = s.list.length;
      if (state.mode === 'quiz' && location.hash === '#quiz') renderResult();
    }
  }, 1000);
  function renderResult() {
    var s = state.session, ok = s.res.filter(function (r) { return r.ok; }).length, n = s.res.length;
    s.done = true;
    if (!s.end) s.end = Date.now();
    var wrong = s.res.filter(function (r) { return !r.ok; }).map(function (r) { return QBY[r.id]; });
    var terms = {};
    wrong.forEach(function (q) { terms[q.term] = true; });
    var h = '<div class="article screen"><h1 class="screen-h">' + esc(s.title) + 'の結果</h1>' +
      '<div class="score"><b>' + (n ? Math.round(100 * ok / n) : 0) + '%</b><span>' + ok + ' / ' + n + ' 問正解・かかった時間 ' + fmtTime(s.end - s.start) + '</span></div>' +
      (s.timeUp ? '<p class="msg ng">時間切れ：' + s.timeUp + ' 問が未回答でした（正答率は回答した問題で計算）。</p>' : '');
    if (wrong.length) {
      h += '<h2 class="home-h">間違えた問題</h2><ul class="wrong-list">' + wrong.map(function (q) {
        return '<li><div class="lazy-math wrong-q">' + q.q + '</div>' + termBtn(q.term) + '</li>';
      }).join('') + '</ul>' +
        '<h2 class="home-h">関係する用語を読み直す</h2><div class="related-list">' + Object.keys(terms).map(function (id) { return termBtn(id); }).join('') + '</div>' +
        '<button type="button" class="big-btn" data-retry-wrong>間違えた問題だけもう一度</button>';
    } else h += '<p>全問正解です。</p>';
    h += s.kind === 'quiz' ? '<button type="button" class="big-btn ghost" data-again>同じ条件で別の問題を解く</button><a class="big-btn ghost" href="#quiz" data-setup>範囲を選び直す</a>'
      : '<a class="big-btn ghost" href="#quiz" data-setup>問題演習へ</a>';
    h += '</div>';
    clearDetail();
    detailEl.innerHTML = h;
    detailEl.scrollTop = 0;
    showMathAll(detailEl.querySelectorAll('.lazy-math'));
  }

  // ---------- 用語カード（用語名を見て意味を思い出す） ----------
  function cardPool() {
    var weak = {};
    if (quizCfg.scope === 'weak') weakTerms().forEach(function (id) { weak[id] = true; });
    return ORDER.filter(function (id) {
      if (!isWritten(id)) return false;
      var ch = PLAN[id].chapter;
      switch (quizCfg.scope) {
        case 'ch': return ch === Number(quizCfg.ch);
        case 'part': var p = PARTS[quizCfg.part] || PARTS[0]; return ch >= p[1] && ch <= p[2];
        case 'weak': return !!weak[id];
        case 'new': return !DB.memo[id];
        case 'bm': return !!DB.bm[id];
        default: return true;
      }
    });
  }
  function startCards() {
    var pool = shuffle(cardPool());
    if (!pool.length) return;
    state.cards = { list: pool.slice(0, quizCfg.count), i: 0, known: 0, again: [] };
    location.hash === '#cards' ? renderCards() : go('#cards');
  }
  function summaryHtml(id) {
    var m = /<section class="sec" data-sec="summary"[^>]*><h2>[\s\S]*?<\/h2>([\s\S]*?)<\/section>/.exec(TERMS[id].html);
    return m ? m[1] : '';
  }
  function renderCards() {
    clearDetail();
    var cs = state.cards;
    if (!cs) {
      detailEl.innerHTML = '<div class="article screen"><h1 class="screen-h">用語カード</h1><p class="screen-sub">用語名を見て意味を思い出し、めくって確かめます。「覚えた」を押すと、その用語に「覚えた」チェックが付きます。範囲と枚数は問題演習と同じ設定を使います。</p>' +
        '<p class="screen-sub">この範囲の用語：<b>' + cardPool().length + '</b> 語</p><button type="button" class="big-btn" data-cards' + (cardPool().length ? '' : ' disabled') + '>始める</button><a class="big-btn ghost" href="#quiz">範囲を選ぶ</a></div>';
      return;
    }
    if (cs.i >= cs.list.length) {
      detailEl.innerHTML = '<div class="article screen"><h1 class="screen-h">用語カードの結果</h1><div class="score"><b>' + cs.known + ' / ' + cs.list.length + '</b><span>覚えた</span></div>' +
        (cs.again.length ? '<h2 class="home-h">まだの用語</h2><div class="related-list">' + cs.again.map(function (id) { return termBtn(id); }).join('') + '</div><button type="button" class="big-btn" data-cards-again>まだの用語だけもう一度</button>' : '') +
        '<button type="button" class="big-btn ghost" data-cards>別の用語でもう一度</button><a class="big-btn ghost" href="#quiz">範囲を選び直す</a></div>';
      detailEl.scrollTop = 0;
      return;
    }
    var id = cs.list[cs.i], t = TERMS[id];
    detailEl.innerHTML = '<div class="article screen"><div class="sess-head"><b>用語カード</b><span class="sess-n">' + (cs.i + 1) + ' / ' + cs.list.length + '</span></div>' +
      '<div class="sess-bar" aria-hidden="true"><span style="width:' + (100 * cs.i / cs.list.length) + '%"></span></div>' +
      '<div class="fc"><p class="crumb">第' + t.chapter + '章 ' + esc(chapterOf(t.chapter).title) + '</p><h2 class="fc-title">' + esc(t.title) + impMark(t.importance) + '</h2>' +
      (t.en ? '<p class="term-meta"><span class="en" lang="en">' + esc(t.en) + '</span></p>' : '') +
      '<div class="fc-back body" hidden>' + summaryHtml(id) + '<p><button type="button" class="rel-btn" data-go="' + id + '">解説を読む →</button></p></div>' +
      '<button type="button" class="big-btn" data-card-flip>めくる</button>' +
      '<div class="fc-judge q-row" hidden><button type="button" class="q-btn primary" data-card-judge="1">覚えた</button><button type="button" class="q-btn" data-card-judge="0">まだ</button></div></div></div>';
    detailEl.scrollTop = 0;
    showMath(detailEl.querySelector('.fc'));
  }

  function renderReview() {
    clearDetail();
    var due = dueList(), nd = nextDue();
    var wrongN = due.filter(function (q) { return DB.ans[q.id].r === 0; }).length;
    var h = '<div class="article screen"><h1 class="screen-h">今日の復習</h1>' +
      '<p class="screen-sub">間違えた問題と、正解してから時間が経った問題を出します。正解が続くほど 1日→3日→7日→14日→30日 と間隔が空きます。</p>';
    if (due.length) {
      h += '<div class="score"><b>' + due.length + '問</b><span>うち間違えたままの問題 ' + wrongN + '問</span></div>' +
        '<button type="button" class="big-btn" data-review>' + Math.min(due.length, 20) + '問を復習する</button>';
      if (state.session && state.session.kind === 'review' && state.session.i < state.session.list.length) h += '<button type="button" class="big-btn ghost" data-resume>解いている途中の復習に戻る（' + (state.session.i + 1) + '/' + state.session.list.length + '）</button>';
    } else {
      h += '<div class="score"><b>0問</b><span>' + (isFinite(nd) ? '次の復習は ' + new Date(nd).toLocaleDateString('ja-JP', { month: 'long', day: 'numeric' }) + ' からです' : 'まだ解いた問題がありません') + '</span></div>' +
        '<a class="big-btn" href="#quiz">問題演習で新しい問題を解く</a>';
    }
    var weak = weakTerms();
    if (weak.length) h += '<h2 class="home-h">苦手な用語</h2><div class="related-list">' + weak.slice(0, 30).map(function (id) { return termBtn(id); }).join('') + '</div>';
    h += '</div>';
    detailEl.innerHTML = h;
    detailEl.scrollTop = 0;
  }

  // ---------- 学習記録・設定 ----------
  function renderStats() {
    clearDetail();
    var today = DB.days[todayKey()] || { a: 0, c: 0, v: 0 };
    var memoN = Object.keys(DB.memo).filter(function (id) { return PLAN[id]; }).length;
    var allA = 0, allC = 0;
    Object.keys(DB.ans).forEach(function (k) { if (QBY[k]) { allA += DB.ans[k].n; allC += DB.ans[k].c; } });
    var h = '<div class="article screen"><h1 class="screen-h">学習記録</h1>' +
      '<div class="tiles">' +
      '<div class="tile"><small>今日の回答</small><b>' + today.a + '</b><span>正解 ' + today.c + '</span></div>' +
      '<div class="tile"><small>連続学習</small><b>' + streakDays() + '日</b><span>問題か用語を見た日</span></div>' +
      '<div class="tile"><small>覚えた用語</small><b>' + memoN + '</b><span>／' + ORDER.length + ' 語</span></div>' +
      '<div class="tile"><small>正答率</small><b>' + (allA ? Math.round(100 * allC / allA) + '%' : '—') + '</b><span>のべ ' + allA + ' 回答</span></div>' +
      '</div>';
    // 直近14日の回答数
    var days = [], mx = 1;
    for (var i = 13; i >= 0; i--) { var k = todayKey(Date.now() - i * DAY), d = DB.days[k] || { a: 0, c: 0, v: 0 }; days.push([k, d]); mx = Math.max(mx, d.a); }
    h += '<h2 class="home-h">直近14日の回答数</h2><div class="spark" role="img" aria-label="直近14日の回答数">' + days.map(function (x) {
      return '<span title="' + x[0] + '：回答 ' + x[1].a + '（正解 ' + x[1].c + '）"><i style="height:' + Math.round(100 * x[1].a / mx) + '%"></i><em>' + Number(x[0].slice(8)) + '</em></span>';
    }).join('') + '</div>';
    // 章ごと
    h += '<h2 class="home-h">章ごとの記録</h2><div class="table-wrap"><table class="ch-table"><thead><tr><th>章</th><th>覚えた率</th><th>正答率</th><th>回答済み</th></tr></thead><tbody>';
    CHAPTERS.forEach(function (c) {
      var ids = c.terms.map(function (t) { return t.id; }), m = ids.filter(function (id) { return DB.memo[id]; }).length;
      var qs = QS.filter(function (q) { return PLAN[q.term] && PLAN[q.term].chapter === c.no; }), n = 0, cc = 0, done = 0;
      qs.forEach(function (q) { var a = DB.ans[q.id]; if (a) { n += a.n; cc += a.c; done++; } });
      var mr = ids.length ? m / ids.length : 0;
      h += '<tr><td><button type="button" class="link-btn" data-goch="' + c.no + '">' + pad2(c.no) + ' ' + esc(c.title) + '</button></td>' +
        '<td><span class="bar"><i style="width:' + Math.round(100 * mr) + '%"></i></span>' + Math.round(100 * mr) + '%</td>' +
        '<td>' + (n ? '<span class="bar"><i style="width:' + Math.round(100 * cc / n) + '%"></i></span>' + Math.round(100 * cc / n) + '%' : '—') + '</td>' +
        '<td>' + done + '/' + qs.length + '</td></tr>';
    });
    h += '</tbody></table></div>';
    // 最近間違えた問題（間違えたままのもの、新しい順）
    var wrongQs = QS.filter(function (q) { var a = DB.ans[q.id]; return a && a.r === 0; })
      .sort(function (x, y) { return DB.ans[y.id].t - DB.ans[x.id].t; }).slice(0, 10);
    if (wrongQs.length) {
      h += '<h2 class="home-h">最近間違えた問題</h2><ul class="wrong-list">' + wrongQs.map(function (q) {
        return '<li><div class="lazy-math wrong-q">' + q.q + '</div>' + termBtn(q.term) + '</li>';
      }).join('') + '</ul><a class="big-btn ghost" href="#review">復習で解き直す</a>';
    }
    var weak = weakTerms();
    h += '<h2 class="home-h">苦手な用語</h2>' + (weak.length ? '<div class="related-list">' + weak.slice(0, 40).map(function (id) { return termBtn(id); }).join('') + '</div>' : '<p class="screen-sub">まだありません。問題を解いて間違えた用語がここに並びます。</p>');
    var bms = Object.keys(DB.bm).filter(isWritten).sort(function (a, b) { return DB.bm[b] - DB.bm[a]; });
    h += '<h2 class="home-h">ブックマーク（' + bms.length + '）</h2>' + (bms.length ? '<div class="related-list">' + bms.map(function (id) { return termBtn(id); }).join('') + '</div>' : '<p class="screen-sub">用語ページの「★ブックマーク」で追加できます。</p>');
    var qbms = Object.keys(DB.qbm).filter(function (id) { return QBY[id]; });
    if (qbms.length) h += '<p class="screen-sub">★を付けた問題：' + qbms.length + ' 問（問題演習の範囲「ブックマーク」で解けます）</p>';
    var notes = Object.keys(DB.notes).filter(isWritten).sort(function (a, b) { return DB.notes[b].t - DB.notes[a].t; });
    if (notes.length) h += '<h2 class="home-h">メモを書いた用語（' + notes.length + '）</h2><div class="related-list">' + notes.map(function (id) { return termBtn(id); }).join('') + '</div>';
    // 設定
    h += '<h2 class="home-h">設定</h2><fieldset class="seg"><legend>文字サイズ</legend><div class="seg-row">' + [['s', '小'], ['m', '中'], ['l', '大']].map(function (f) {
      return '<button type="button" class="chip" data-font="' + f[0] + '" aria-pressed="' + (DB.font === f[0]) + '">' + f[1] + '</button>';
    }).join('') + '</div></fieldset>' +
      '<fieldset class="seg"><legend>試験日（ホームに残り日数を出す）</legend><div class="seg-row"><input type="date" class="date-in" data-exam value="' + esc(DB.exam) + '" aria-label="試験日">' +
      (DB.exam ? '<button type="button" class="chip" data-exam-clear>消す</button>' : '') + '</div></fieldset>' +
      '<fieldset class="seg"><legend>学習データ（iPad とスマホの間で移す）</legend>' +
      '<p class="screen-sub">「覚えた」チェック、ブックマーク、メモ、問題の回答履歴をファイルに書き出し、別の端末で読み込めます。</p>' +
      '<div class="seg-row"><button type="button" class="chip" data-export>ファイルに書き出す</button><label class="chip file-chip">ファイルから読み込む<input type="file" accept="application/json,.json" data-import hidden></label></div>' +
      '<div class="import-box" hidden></div></fieldset>' +
      '</div>';
    detailEl.innerHTML = h;
    detailEl.scrollTop = 0;
    showMathAll(detailEl.querySelectorAll('.lazy-math'));
  }

  function setFont(f) {
    DB.font = f; persist(true);
    document.documentElement.setAttribute('data-font', f);
    if (window.Charts) Charts.redrawAll();
  }
  function exportData() {
    var payload = { app: 'jun1-dictionary', v: DB_VERSION, exported: new Date().toISOString(), data: DB };
    var blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'jun1-dictionary-' + todayKey() + '.json';
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }
  // 2つの記録を合わせる（新しい方・多い方を残す）
  function mergeDB(a, b) {
    var o = sanitize(JSON.parse(JSON.stringify(a)));
    ['memo', 'bm', 'qbm'].forEach(function (k) { Object.keys(b[k]).forEach(function (id) { o[k][id] = Math.max(o[k][id] || 0, b[k][id]); }); });
    Object.keys(b.notes).forEach(function (id) {
      var x = o.notes[id], y = b.notes[id];
      if (!x || y.t > x.t) o.notes[id] = y;
    });
    Object.keys(b.ans).forEach(function (id) {
      var x = o.ans[id], y = b.ans[id];
      if (!x) { o.ans[id] = y; return; }
      var newer = y.t > x.t ? y : x;
      o.ans[id] = { n: Math.max(x.n, y.n), c: Math.max(x.c, y.c), s: newer.s, t: newer.t, d: newer.d, r: newer.r };
    });
    Object.keys(b.days).forEach(function (k) {
      var x = o.days[k] || { a: 0, c: 0, v: 0 }, y = b.days[k];
      o.days[k] = { a: Math.max(x.a, y.a), c: Math.max(x.c, y.c), v: Math.max(x.v, y.v) };
    });
    if (!o.exam && b.exam) o.exam = b.exam;
    var hs = {};
    o.mock.hist.forEach(function (h) { hs[h.s + '-' + h.t] = 1; });
    b.mock.hist.forEach(function (h) { if (!hs[h.s + '-' + h.t]) o.mock.hist.push(h); });
    o.mock.hist.sort(function (x, y) { return x.t - y.t; });
    if (!o.mock.cur && b.mock.cur) o.mock.cur = b.mock.cur;
    var seen = {};
    o.recent = o.recent.concat(b.recent).filter(function (id) { if (seen[id]) return false; seen[id] = 1; return true; }).slice(0, 30);
    return o;
  }
  function importFile(file) {
    var box = detailEl.querySelector('.import-box');
    var rd = new FileReader();
    rd.onload = function () {
      var d = null;
      try { d = JSON.parse(rd.result); } catch (e) { d = null; }
      if (!isObj(d) || d.app !== 'jun1-dictionary' || !isObj(d.data)) {
        box.hidden = false; box.innerHTML = '<p class="msg ng">この辞典の学習データのファイルではないようです。</p>'; return;
      }
      if (num(d.v, 0) > DB_VERSION) { box.hidden = false; box.innerHTML = '<p class="msg ng">新しい版で書き出されたファイルです。ページを再読み込みしてから試してください。</p>'; return; }
      var inc = sanitize(d.data);
      box.hidden = false;
      box.innerHTML = '<p>読み込むデータ：覚えた ' + Object.keys(inc.memo).length + ' 語、ブックマーク ' + Object.keys(inc.bm).length + ' 語、メモ ' + Object.keys(inc.notes).length + ' 件、回答した問題 ' + Object.keys(inc.ans).length + ' 問、模擬試験の受験 ' + inc.mock.hist.length + ' 回' + (inc.mock.cur ? '（途中のセットあり）' : '') + '（' + esc(String(d.exported || '').slice(0, 10)) + ' に書き出し）</p>' +
        '<div class="seg-row"><button type="button" class="chip" data-imp="merge">今の記録と統合する</button><button type="button" class="chip" data-imp="replace">今の記録を上書きする</button><button type="button" class="chip" data-imp="cancel">やめる</button></div>';
      box.__inc = inc;
    };
    rd.readAsText(file);
  }

  // ---------- 模擬試験（本番形式 25問・90分。content/exam_sets → build.py → DATA.exams） ----------
  // 記録は DB.mock = { hist: [受験結果], cur: 解いている途中のセット }。
  // 本番モードの残り時間は「画面を開いている間だけ」減る（閉じても続きから再開できるように、残り時間を保存する）
  var EXAMS = DATA.exams || [], EXBY = {}, EX_SETS = (SITE.exam && SITE.exam.sets) || 30, PASS = 60;
  EXAMS.forEach(function (e) { e.items.forEach(function (it, i) { it.n = i + 1; }); EXBY[e.no] = e; });
  // 途中経過を保存したあとで問題の版（選択肢の並びなど）が変わったら、選んだ番号がずれるので、その途中経過は使わない
  (function () {
    var c = DB.mock.cur, e = c && EXBY[c.s];
    if (e && c.v !== (e.v || '')) { DB.mock.cur = null; persist(true); }
  })();
  var CIRC = ['①', '②', '③', '④', '⑤'];
  function exCorrect(it, v) {
    if (v === undefined || v === null || v === '') return false;
    if (it.type === 'choice') return v === it.a;
    var x = parseNum(v);
    return !isNaN(x) && Math.abs(x - it.a) <= it.tol + 1e-9;
  }
  function exHist(no) { return DB.mock.hist.filter(function (h) { return h.s === no; }); }
  function exBest(no) { var b = null; exHist(no).forEach(function (h) { if (!b || h.sc > b.sc) b = h; }); return b; }
  function fmtMin(ms) { var t = Math.max(0, Math.round(ms / 1000)); return Math.floor(t / 60) + ':' + String(t % 60).padStart(2, '0'); }
  function chTitle(no) { var c = chapterOf(no); return '第' + no + '章 ' + (c ? c.title : ''); }
  // 1問の配点（2級は 100÷35 なので小数第1位までに丸めて「約」を付ける）
  function ptsLabel(e) { return e.pts === Math.round(e.pts) ? String(e.pts) : '約' + Math.round(e.pts * 10) / 10; }
  function passTag(sc) { return sc >= PASS ? '<span class="ex-pass ok">合格</span>' : '<span class="ex-pass ng">不合格</span>'; }

  // 受験履歴の得点の推移（1系列の折れ線＋合格ライン。点にポインタを当てると内容が出る）
  function exChart(hist) {
    if (hist.length < 2) return '';
    var W = 640, H = 220, L = 34, R = 12, T = 14, B = 28, n = hist.length;
    var x = function (i) { return L + (W - L - R) * (n === 1 ? 0.5 : i / (n - 1)); };
    var y = function (v) { return T + (H - T - B) * (1 - v / 100); };
    var s = '<svg class="ex-chart" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="模擬試験の得点の推移">';
    [0, 20, 40, 60, 80, 100].forEach(function (v) {
      s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + y(v) + '" y2="' + y(v) + '" class="' + (v === PASS ? 'ex-pl' : 'ex-grid') + '"/>' +
        '<text x="' + (L - 6) + '" y="' + (y(v) + 4) + '" text-anchor="end" class="ex-ax">' + v + '</text>';
    });
    s += '<text x="' + (W - R) + '" y="' + (y(PASS) - 5) + '" text-anchor="end" class="ex-ax ex-pt">合格ライン 60</text>';
    s += '<polyline fill="none" class="ex-line" points="' + hist.map(function (h, i) { return x(i) + ',' + y(h.sc); }).join(' ') + '"/>';
    hist.forEach(function (h, i) {
      var d = new Date(h.t);
      s += '<g class="ex-dot"><circle cx="' + x(i) + '" cy="' + y(h.sc) + '" r="12" class="ex-hit"/><circle cx="' + x(i) + '" cy="' + y(h.sc) + '" r="4.5"/>' +
        '<title>' + (d.getMonth() + 1) + '/' + d.getDate() + ' 第' + h.s + '回（' + (h.m === 'p' ? '練習' : '本番') + '）' + h.sc + '点</title></g>';
    });
    s += '<text x="' + L + '" y="' + (H - 8) + '" class="ex-ax">1回目</text><text x="' + (W - R) + '" y="' + (H - 8) + '" text-anchor="end" class="ex-ax">' + n + '回目</text></svg>';
    return s;
  }
  // 章ごとの正答率（全受験の合計）。苦手な章＝正答率の低い章
  function exChStats() {
    var ch = {};
    DB.mock.hist.forEach(function (h) { Object.keys(h.ch).forEach(function (k) { var a = ch[k] || (ch[k] = [0, 0]); a[0] += h.ch[k][0]; a[1] += h.ch[k][1]; }); });
    return Object.keys(ch).map(function (k) { return { ch: Number(k), c: ch[k][0], n: ch[k][1], r: ch[k][0] / ch[k][1] }; });
  }

  function renderExamList() {
    clearDetail();
    var cur = DB.mock.cur, h = '<div class="article screen"><h1 class="screen-h">模擬試験</h1>' +
      '<p class="screen-sub">' + (SITE.exam && SITE.exam.desc ? SITE.exam.desc : '本番と同じ <b>25問・90分</b>（5肢選択と数値入力、各4点、<b>60点以上で合格</b>）。電卓（四則・平方根）とメモ用紙を用意して解きます。') +
      '公開しているのは、自動検査・別の方法での検算・見直しの3段階を通った問題だけです。</p>';
    if (cur && EXBY[cur.s]) {
      var nAns = Object.keys(cur.a).length;
      h += '<div class="ex-resume"><div><b>第' + cur.s + '回（' + (cur.m === 'p' ? '練習' : '本番') + 'モード）の途中です</b><p>' + nAns + ' / ' + EXBY[cur.s].items.length + ' 問に解答' +
        (cur.m === 'r' ? '・残り ' + fmtMin(cur.left) : '') + '</p></div><a class="q-btn primary" href="#exam/' + cur.s + '/run">続きから解く</a></div>';
    }
    h += '<ol class="ex-sets">';
    for (var no = 1; no <= EX_SETS; no++) {
      var e = EXBY[no], hs = exHist(no), best = exBest(no), last = hs[hs.length - 1];
      if (!e) { h += '<li class="ex-set off"><b>第' + no + '回</b><span>準備中</span></li>'; continue; }
      h += '<li class="ex-set"><a href="#exam/' + no + '"><b>第' + no + '回</b>' +
        (last ? '<span>最新 ' + last.sc + '点 ' + passTag(last.sc) + '</span><span class="ex-sub">最高 ' + best.sc + '点・' + hs.length + '回受験</span>' : '<span class="ex-sub">未受験</span>') + '</a></li>';
    }
    h += '</ol>';
    var hist = DB.mock.hist.slice().sort(function (a, b) { return a.t - b.t; });
    if (hist.length) {
      h += '<h2 class="home-h">得点の推移</h2>' + (exChart(hist) || '<p class="screen-sub">2回以上受けるとグラフが出ます。</p>');
      var ws = exChStats().filter(function (x) { return x.n >= 1; }).sort(function (a, b) { return a.r - b.r || b.n - a.n; }).slice(0, 5);
      h += '<h2 class="home-h">苦手な章（正答率の低い順）</h2><ul class="ex-weak">' + ws.map(function (x) {
        return '<li><a href="#ch' + pad2(x.ch) + '">' + esc(chTitle(x.ch)) + '</a><span class="ex-bar"><i style="width:' + Math.round(100 * x.r) + '%"></i></span><span>' + x.c + '/' + x.n + '（' + Math.round(100 * x.r) + '%）</span></li>';
      }).join('') + '</ul><p><a class="q-btn ghost" href="#roadmap">ロードマップで前提の用語を確かめる</a></p>';
      h += '<h2 class="home-h">受験履歴</h2><div class="table-wrap"><table class="ex-table"><thead><tr><th>日時</th><th>回</th><th>モード</th><th>得点</th><th>時間</th><th></th></tr></thead><tbody>' +
        hist.slice().reverse().map(function (x) {
          var d = new Date(x.t);
          return '<tr><td>' + (d.getMonth() + 1) + '/' + d.getDate() + ' ' + pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + '</td><td>第' + x.s + '回</td><td>' + (x.m === 'p' ? '練習' : '本番') + '</td><td>' + x.sc + '点 ' + passTag(x.sc) +
            '</td><td>' + fmtTime(x.d) + '</td><td><a href="#exam/' + x.s + '/result/' + x.t + '">結果</a></td></tr>';
        }).join('') + '</tbody></table></div>';
    }
    h += '</div>';
    detailEl.innerHTML = h;
    detailEl.scrollTop = 0;
  }

  function renderExamStart(no) {
    clearDetail();
    var e = EXBY[no];
    if (!e) { detailEl.innerHTML = '<div class="article screen"><h1 class="screen-h">模擬試験 第' + no + '回</h1><p>このセットは準備中です。</p><a class="q-btn" href="#exam">一覧へ</a></div>'; return; }
    var chs = {};
    e.items.forEach(function (it) { chs[it.ch] = (chs[it.ch] || 0) + 1; });
    var cur = DB.mock.cur, busy = cur && cur.s !== no;
    var lv = { '基本': 0, '標準': 0, '応用': 0 };
    e.items.forEach(function (it) { lv[it.lv]++; });
    detailEl.innerHTML = '<div class="article screen"><p class="crumb"><a href="#exam">模擬試験</a></p><h1 class="screen-h">' + esc(e.title) + '</h1>' +
      '<p class="screen-sub">' + e.items.length + '問・' + e.min + '分・各' + ptsLabel(e) + '点（大問 ' + e.probs.length + '）。難易度 基本 ' + lv['基本'] + '・標準 ' + lv['標準'] + '・応用 ' + lv['応用'] + '。</p>' +
      '<p class="screen-sub">出題範囲：' + Object.keys(chs).map(function (k) { return '第' + k + '章'; }).join('・') + '</p>' +
      (cur && cur.s === no ? '<a class="big-btn" href="#exam/' + no + '/run">途中から再開する</a>' : '') +
      '<div class="ex-modes"><button type="button" class="ex-mode" data-ex-start="r"><b>本番モード</b><span>残り時間のタイマー。問題番号から自由に移動し、「あとで見直す」の印を付けられます。最後にまとめて採点し、時間切れで自動採点します。</span></button>' +
      '<button type="button" class="ex-mode" data-ex-start="p"><b>練習モード</b><span>時間制限なし。1問ごとに答え合わせをして、正誤と解説を見ながら進めます。</span></button></div>' +
      (busy ? '<p class="msg ng">第' + cur.s + '回を解いている途中です。始めると、その途中経過は消えます。</p>' : '') +
      (cur && cur.s === no ? '<p class="msg ng">始め直すと、今の途中経過は消えます。</p>' : '') +
      (exHist(no).length ? '<p><a class="q-btn ghost" href="#exam/' + no + '/result">前回の結果を見る</a></p>' : '') + '</div>';
    detailEl.scrollTop = 0;
  }
  function exStart(no, mode) {
    var e = EXBY[no];
    DB.mock.cur = { s: no, m: mode, i: 0, left: e.min * 60000, el: 0, a: {}, f: {}, k: {}, st: Date.now(), v: e.v || '' };
    persist(true);
    if (location.hash === '#exam/' + no + '/run') renderExamRun(); else go('#exam/' + no + '/run');
  }

  function exItemHtml(e, it, cur) {
    var p = e.probs[it.p], v = cur.a[it.id], checked = cur.m === 'p' && cur.k[it.id];
    var first = e.items.filter(function (x) { return x.p === it.p; });
    var h = '<div class="ex-prob"><p class="ex-prob-h">大問' + (it.p + 1) + '　' + esc(p.t) + (first.length > 1 ? '<small>（問' + first[0].n + '〜' + first[first.length - 1].n + '）</small>' : '') + '</p><div class="body ex-ctx">' + p.ctx + '</div></div>' +
      '<h2 class="ex-qh">問' + it.n + '<small>' + ptsLabel(e) + '点・' + esc(it.lv) + '</small></h2><div class="body ex-text">' + it.q + '</div>';
    if (it.type === 'choice') {
      h += '<div class="ex-opts" role="radiogroup">' + it.c.map(function (c, i) {
        var cls = checked ? (i === it.a ? ' correct' : (i === v ? ' wrong' : '')) : '';
        return '<button type="button" class="ex-opt' + cls + '" role="radio" data-ex-opt="' + i + '" aria-checked="' + (v === i) + '"' + (checked ? ' disabled' : '') + '><span class="ex-key">' + CIRC[i] + '</span><span>' + c + '</span></button>';
      }).join('') + '</div>';
    } else {
      h += '<div class="ex-num"><p class="ex-fmt">' + it.fmt + '</p><input type="text" inputmode="decimal" class="q-num ex-in" aria-label="答えの数値" autocomplete="off" value="' + esc(v == null ? '' : v) + '"' + (checked ? ' disabled' : '') + '></div>';
    }
    h += '<div class="q-row"><button type="button" class="q-btn ghost ex-flag" data-ex-flag aria-pressed="' + !!cur.f[it.id] + '">' + (cur.f[it.id] ? '★ あとで見直す（印あり）' : '☆ あとで見直す') + '</button>' +
      (cur.m === 'p' && !checked ? '<button type="button" class="q-btn primary" data-ex-check>答え合わせ</button>' : '') + '</div>';
    if (checked) h += exAnswerBox(it, v);
    return h;
  }
  // 正誤・正解・解説・関係する用語（練習モードの答え合わせと、採点結果で共通）
  function exAnswerBox(it, v) {
    var ok = exCorrect(it, v);
    var yours = v === undefined || v === '' ? '未解答' : (it.type === 'choice' ? CIRC[v] : esc(v));
    var right = it.type === 'choice' ? CIRC[it.a] : esc(fmtNum(it.a)) + '（±' + esc(fmtNum(it.tol)) + ' まで正解）';
    return '<div class="q-result ' + (ok ? 'ok' : 'ng') + '"><b>' + (ok ? '正解' : '不正解') + '</b><span>あなたの答え：' + yours + '　正解：' + right + '</span></div>' +
      '<div class="q-exp ex-exp"><b>解説</b>' + it.e + (it.terms.length ? '<p class="ex-terms">関係する用語：' + it.terms.map(function (id) { return termBtn(id); }).join('') + '</p>' : '') + '</div>';
  }
  function renderExamRun() {
    clearDetail();
    var cur = DB.mock.cur, e = cur && EXBY[cur.s];
    if (!e) { renderExamList(); return; }
    cur.i = Math.max(0, Math.min(e.items.length - 1, cur.i));
    var it = e.items[cur.i];
    var grid = e.items.map(function (x, i) {
      var c = (cur.a[x.id] !== undefined && cur.a[x.id] !== '' ? ' done' : '') + (cur.f[x.id] ? ' flag' : '');
      if (cur.m === 'p' && cur.k[x.id]) c += exCorrect(x, cur.a[x.id]) ? ' ok' : ' ng';
      return '<button type="button" class="ex-g' + c + '" data-ex-go="' + i + '"' + (i === cur.i ? ' aria-current="true"' : '') + ' aria-label="問' + (i + 1) + '">' + (i + 1) + '</button>';
    }).join('');
    var nAns = e.items.filter(function (x) { return cur.a[x.id] !== undefined && cur.a[x.id] !== ''; }).length;
    detailEl.innerHTML = '<div class="article screen ex-run"><div class="sess-head ex-head"><a class="q-btn ghost" href="#exam">中断</a><b>第' + cur.s + '回・' + (cur.m === 'p' ? '練習' : '本番') + '</b>' +
      '<span class="sess-timer ex-timer" aria-live="off">' + (cur.m === 'r' ? '残り ' + fmtMin(cur.left) : '経過 ' + fmtMin(cur.el)) + '</span><span class="sess-n">解答 ' + nAns + '/' + e.items.length + '</span></div>' +
      '<nav class="ex-grid" aria-label="問題番号の一覧（★はあとで見直す）">' + grid + '</nav>' +
      '<div class="ex-item">' + exItemHtml(e, it, cur) + '</div>' +
      '<div class="ex-nav"><button type="button" class="q-btn" data-ex-step="-1"' + (cur.i ? '' : ' disabled') + '>← 前の問題</button><button type="button" class="q-btn" data-ex-step="1"' + (cur.i < e.items.length - 1 ? '' : ' disabled') + '>次の問題 →</button></div>' +
      '<div class="ex-finish"><button type="button" class="big-btn" data-ex-finish>解答を終了して採点する</button></div></div>';
    detailEl.scrollTop = 0;
    showMath(detailEl.querySelector('.ex-item'));
    exTick.last = Date.now();
  }
  function exSaveInput() {
    var cur = DB.mock.cur, inp = detailEl.querySelector('.ex-in');
    if (!cur || !inp || inp.disabled) return;
    var e = EXBY[cur.s], it = e.items[cur.i], v = inp.value.trim().slice(0, 40);
    if (v) cur.a[it.id] = v; else delete cur.a[it.id];
  }
  function exGrade(timeUp) {
    var cur = DB.mock.cur, e = cur && EXBY[cur.s];
    if (!e) return;
    var c = 0, ch = {};
    e.items.forEach(function (it) {
      var ok = exCorrect(it, cur.a[it.id]), a = ch[it.ch] || (ch[it.ch] = [0, 0]);
      a[1]++; if (ok) { c++; a[0]++; }
    });
    var h = { s: cur.s, sc: Math.round(c * e.pts * 10) / 10, c: c, n: e.items.length, t: Date.now(), d: cur.m === 'r' ? e.min * 60000 - Math.max(0, cur.left) : cur.el,
      m: cur.m, ch: ch, a: cur.a, tu: timeUp ? 1 : 0, v: cur.v || '' };
    DB.mock.hist.push(h);
    DB.mock.cur = null;
    persist(true);
    go('#exam/' + h.s + '/result/' + h.t);
  }
  function renderExamResult(no, t) {
    clearDetail();
    var e = EXBY[no], hs = exHist(no), r = null;
    hs.forEach(function (x) { if (!t || x.t === t) r = x; });
    if (!e || !r) { renderExamList(); return; }
    var wrong = e.items.filter(function (it) { return !exCorrect(it, r.a[it.id]); });
    var chs = Object.keys(r.ch).map(Number).sort(function (a, b) { return a - b; });
    var h = '<div class="article screen"><p class="crumb"><a href="#exam">模擬試験</a></p><h1 class="screen-h">' + esc(e.title) + 'の結果</h1>' +
      (r.tu ? '<p class="msg ng">時間切れのため、その時点の解答で自動採点しました。</p>' : '') +
      ((r.v || '') !== (e.v || '') ? '<p class="msg">この結果のあとで問題が更新されました（選択肢の並べ替えなど）。点数は受験したときのものです。下の正誤と解説は今の問題で表示しているため、当時の解答とずれることがあります。</p>' : '') +
      '<div class="score ex-score"><b>' + r.sc + '<small>点</small></b>' + passTag(r.sc) + '<span>' + r.c + ' / ' + r.n + ' 問正解・' + (r.m === 'p' ? '練習' : '本番') + 'モード・' + fmtTime(r.d) + '</span></div>' +
      '<div class="ex-meter" aria-label="得点 ' + r.sc + ' 点（合格ライン 60 点）"><i style="width:' + r.sc + '%" class="' + (r.sc >= PASS ? 'ok' : 'ng') + '"></i><span class="ex-meter-line" style="left:' + PASS + '%"></span></div>' +
      '<p class="screen-sub">' + (r.sc >= PASS ? '合格ライン（60点）を ' + Math.round((r.sc - PASS) * 10) / 10 + ' 点上回りました。' : '合格ライン（60点）まであと ' + Math.round((PASS - r.sc) * 10) / 10 + ' 点（' + Math.ceil((PASS - r.sc) / e.pts) + ' 問）です。') + '</p>' +
      '<h2 class="home-h">章ごとの正答率</h2><div class="table-wrap"><table class="ex-table"><thead><tr><th>章</th><th>正解</th><th>正答率</th></tr></thead><tbody>' +
      chs.map(function (k) { var a = r.ch[k]; return '<tr><td><a href="#ch' + pad2(k) + '">' + esc(chTitle(k)) + '</a></td><td>' + a[0] + '/' + a[1] + '</td><td><span class="ex-bar"><i style="width:' + Math.round(100 * a[0] / a[1]) + '%"></i></span></td></tr>'; }).join('') +
      '</tbody></table></div>';
    h += '<h2 class="home-h">間違えた問題（' + wrong.length + '問）</h2>' + (wrong.length ? '' : '<p>全問正解です。</p>') + '<div class="ex-review">' +
      wrong.map(function (it) { return exReviewHtml(e, it, r); }).join('') + '</div>' +
      '<details class="ex-all"><summary>すべての問題と解説を見る</summary><div class="ex-review">' + e.items.filter(function (it) { return wrong.indexOf(it) < 0; }).map(function (it) { return exReviewHtml(e, it, r); }).join('') + '</div></details>' +
      '<div class="ex-after"><a class="big-btn" href="#exam/' + no + '">もう一度受ける</a><a class="big-btn ghost" href="#roadmap">ロードマップで復習する用語を選ぶ</a><a class="big-btn ghost" href="#exam">模擬試験の一覧</a></div></div>';
    detailEl.innerHTML = h;
    detailEl.scrollTop = 0;
    showMathAll(detailEl.querySelectorAll('.ex-review > .ex-rv'));
  }
  function exReviewHtml(e, it, r) {
    var p = e.probs[it.p];
    return '<article class="ex-rv"><p class="ex-prob-h">問' + it.n + '（大問' + (it.p + 1) + '　' + esc(p.t) + '・' + esc(chTitle(it.ch)) + '）</p>' +
      '<details class="ex-ctx-d"><summary>状況設定を見る</summary><div class="body">' + p.ctx + '</div></details>' +
      '<div class="body">' + it.q + '</div>' +
      (it.type === 'choice' ? '<ol class="ex-olist">' + it.c.map(function (c, i) { return '<li' + (i === it.a ? ' class="ans"' : '') + '><span class="ex-key">' + CIRC[i] + '</span><span>' + c + '</span></li>'; }).join('') + '</ol>' : '<p class="ex-fmt">' + it.fmt + '</p>') +
      exAnswerBox(it, r.a[it.id]) + '</article>';
  }
  // 残り時間（1秒ごと）。画面が見えていて、解いている画面のときだけ進める
  function exTick() {
    var cur = DB.mock.cur, now = Date.now(), dt = now - (exTick.last || now);
    exTick.last = now;
    if (!cur || !/^#exam\/\d+\/run$/.test(location.hash) || document.hidden) return;
    dt = Math.min(dt, 5000);
    var el = detailEl.querySelector('.ex-timer');
    if (cur.m === 'r') {
      cur.left -= dt;
      if (el) { el.textContent = '残り ' + fmtMin(cur.left); el.classList.toggle('low', cur.left < 5 * 60000); }
      if (cur.left <= 0) { exSaveInput(); exGrade(true); return; }
    } else {
      cur.el += dt;
      if (el) el.textContent = '経過 ' + fmtMin(cur.el);
    }
    if (++exTick.n % 5 === 0) persist();
  }
  exTick.n = 0;
  setInterval(exTick, 1000);
  document.addEventListener('visibilitychange', function () { exTick.last = Date.now(); if (document.hidden && DB.mock.cur) { exSaveInput(); persist(true); } });

  detailEl.addEventListener('click', function (ev) {
    var b = ev.target.closest('button');
    if (!b || !detailEl.querySelector('.ex-run, .ex-modes')) return;
    var cur = DB.mock.cur;
    if (b.dataset.exStart) { var no = Number(/^#exam\/(\d+)/.exec(location.hash)[1]); exStart(no, b.dataset.exStart); return; }
    if (!cur || !EXBY[cur.s]) return;
    var e = EXBY[cur.s], it = e.items[cur.i];
    if (b.dataset.exOpt !== undefined) {
      cur.a[it.id] = Number(b.dataset.exOpt);
      Array.prototype.forEach.call(detailEl.querySelectorAll('.ex-opt'), function (o) { o.setAttribute('aria-checked', String(o === b)); });
      var g = detailEl.querySelector('.ex-g[data-ex-go="' + cur.i + '"]'); if (g) g.classList.add('done');
      persist();
      return;
    }
    if (b.hasAttribute('data-ex-flag')) {
      exSaveInput();
      if (cur.f[it.id]) delete cur.f[it.id]; else cur.f[it.id] = 1;
      persist(); renderExamRun(); return;
    }
    if (b.hasAttribute('data-ex-check')) {
      exSaveInput();
      if (cur.a[it.id] === undefined) { var inp = detailEl.querySelector('.ex-in'); if (inp) { inp.focus(); inp.classList.add('bad'); } return; }
      cur.k[it.id] = 1; persist(); renderExamRun(); return;
    }
    if (b.dataset.exGo !== undefined) { exSaveInput(); cur.i = Number(b.dataset.exGo); persist(); renderExamRun(); return; }
    if (b.dataset.exStep) { exSaveInput(); cur.i += Number(b.dataset.exStep); persist(); renderExamRun(); return; }
    if (b.hasAttribute('data-ex-finish')) {
      exSaveInput();
      var left = e.items.filter(function (x) { return cur.a[x.id] === undefined; }).length, fl = Object.keys(cur.f).length;
      if ((left || fl) && !window.confirm((left ? '未解答が ' + left + ' 問あります。' : '') + (fl ? '「あとで見直す」の印が ' + fl + ' 問にあります。' : '') + '採点しますか？')) return;
      exGrade(false);
    }
  });
  detailEl.addEventListener('input', function (ev) {
    if (!ev.target.classList.contains('ex-in')) return;
    ev.target.classList.remove('bad');
    exSaveInput();
    var cur = DB.mock.cur, g = cur && detailEl.querySelector('.ex-g[data-ex-go="' + cur.i + '"]');
    if (g) g.classList.toggle('done', !!ev.target.value.trim());
    persist();
  });

  // ---------- ロードマップ（content/roadmap.yaml → DATA.roadmap） ----------
  var RMAP = DATA.roadmap;
  function termAcc(id) {
    var c = 0, n = 0;
    (QTERM[id] || []).forEach(function (q) { var a = DB.ans[q.id]; if (a) { c += a.c; n += a.n; } });
    return { c: c, n: n };
  }
  function rmNext(k) {
    if (!RMAP) return [];
    return RMAP.terms.filter(function (t) {
      return isWritten(t.id) && !DB.memo[t.id] && t.pre.every(function (p) { return DB.memo[p] || !isWritten(p); });
    }).sort(function (a, b) { return a.st - b.st || b.sc - a.sc; }).slice(0, k);
  }
  function renderRoadmap() {
    clearDetail();
    if (!RMAP) { detailEl.innerHTML = '<div class="article screen"><h1 class="screen-h">ロードマップ</h1><p>準備中です。</p></div>'; return; }
    var h = '<div class="article screen rm"><h1 class="screen-h">学習ロードマップ</h1>' +
      '<p class="screen-sub">出題されやすさ・得点への効きやすさ・ほかの用語の前提になるかで優先度（0〜100）を付け、前提の用語が先に来るように並べました。用語を読んで「覚えた」にチェックすると進み具合に入ります。</p>';
    var nx = rmNext(5);
    h += '<section class="rm-next"><h2 class="home-h">次にやる5語</h2>' + (nx.length ? '<ol>' + nx.map(function (t) {
      return '<li><button type="button" class="rel-btn strong" data-go="' + t.id + '">' + esc(PLAN[t.id].title) + '</button><span class="rm-why">' + esc(t.why) + '（目安 ' + t.min + '分）</span></li>';
    }).join('') + '</ol>' : '<p>前提を満たした未習得の用語はありません。</p>') + '</section>';
    var firstOpen = -1;
    RMAP.stages.forEach(function (s, si) {
      var ts = RMAP.terms.filter(function (t) { return t.st === s.no; });
      var done = ts.filter(function (t) { return DB.memo[t.id]; }).length, pct = ts.length ? Math.round(100 * done / ts.length) : 0;
      var mins = ts.reduce(function (a, t) { return a + (t.min || 0); }, 0);
      if (firstOpen < 0 && done < ts.length) firstOpen = si;
      h += '<details class="rm-stage"' + (firstOpen === si ? ' open' : '') + '><summary><span class="rm-st-h"><b>第' + s.no + '段階：' + esc(s.title) + '</b><small>' + ts.length + '語・目安 ' + Math.round(mins / 60) + '時間</small></span>' +
        '<span class="ex-bar rm-bar"><i style="width:' + pct + '%"></i></span><span class="rm-pct">' + done + '/' + ts.length + '（' + pct + '%）</span></summary>' +
        (s.desc ? '<p class="screen-sub">' + esc(s.desc) + '</p>' : '') + '<ol class="rm-list">' + ts.map(function (t) {
          var acc = termAcc(t.id), p = PLAN[t.id];
          var pre = t.pre.filter(function (x) { return PLAN[x]; });
          return '<li class="rm-row' + (DB.memo[t.id] ? ' done' : '') + '"><button type="button" class="memo-btn" data-rm-memo="' + t.id + '" aria-pressed="' + !!DB.memo[t.id] + '" aria-label="覚えた">' + (DB.memo[t.id] ? CHECK_SVG : CHECK_OFF) + '</button>' +
            '<div><div class="rm-t">' + (isWritten(t.id) ? '<button type="button" class="rm-link" data-go="' + t.id + '">' + esc(p.title) + '</button>' : esc(p.title)) + impMark(p.importance) +
            '<span class="rm-sc" title="優先度">' + t.sc + '</span></div><p class="rm-why">' + esc(t.why) + '</p>' +
            '<p class="rm-meta">第' + p.chapter + '章・目安 ' + t.min + '分・問題 ' + (acc.n ? acc.c + '/' + acc.n + '（' + Math.round(100 * acc.c / acc.n) + '%）' : '未回答') +
            (pre.length ? '・前提：' + pre.map(function (x) { return (DB.memo[x] ? '✓' : '') + esc(PLAN[x].title); }).join('、') : '') + '</p></div></li>';
        }).join('') + '</ol></details>';
    });
    h += '</div>';
    detailEl.innerHTML = h;
    detailEl.scrollTop = 0;
  }
  detailEl.addEventListener('click', function (ev) {
    var b = ev.target.closest('[data-rm-memo]');
    if (!b) return;
    var id = b.dataset.rmMemo;
    if (DB.memo[id]) delete DB.memo[id]; else DB.memo[id] = Date.now();
    persist(true);
    var y = detailEl.scrollTop;
    renderRoadmap();
    detailEl.scrollTop = y;
    renderList();
  });

  // ---------- 公式まとめ ----------
  function renderFormulas() {
    clearDetail();
    detailEl.innerHTML = '<button type="button" class="back-btn" data-back>← 一覧</button><div class="article formulas"><h1 class="screen-h">公式まとめ</h1>' +
      (FORMULAS ? '<nav class="jump" aria-label="節へ移動"></nav><div class="body">' + FORMULAS + '</div>' : '<p>準備中です。</p>') + '</div>';
    var jump = detailEl.querySelector('.jump');
    if (jump) {
      jump.innerHTML = Array.prototype.map.call(detailEl.querySelectorAll('.body > .sec'), function (s) {
        return '<button type="button" data-jump="' + s.dataset.sec + '">' + esc(s.querySelector('h2').textContent.replace(/^第\d+章\s*/, '')) + '</button>';
      }).join('');
    }
    detailEl.scrollTop = 0;
    showMathAll(detailEl.querySelectorAll('.sec'));
    showMath(detailEl);   // 最初の見出しより前の「記号の約束」なども組む
  }

  // ---------- 解説側のクリック ----------
  detailEl.addEventListener('click', function (e) {
    var b = e.target.closest('button, a');
    if (!b) return;
    if (b.hasAttribute('data-back')) { backToList(); return; }
    if (b.dataset.goch) { go('#ch' + pad2(b.dataset.goch)); return; }
    if (b.dataset.go !== undefined && b.dataset.go !== '') { state.fromList = false; go('#' + b.dataset.go); return; }
    if (b.dataset.jump) {
      var s = detailEl.querySelector('.sec[data-sec="' + b.dataset.jump + '"]');
      if (s) s.scrollIntoView({ block: 'start', behavior: 'smooth' });
      return;
    }
    if (b.dataset.stepQuiz !== undefined) { startStepQuiz(Number(b.dataset.stepQuiz)); return; }
    if (b.dataset.stepCards !== undefined) { startStepCards(Number(b.dataset.stepCards)); return; }
    if (b.hasAttribute('data-random')) { var w = ORDER.filter(isWritten); go('#' + w[Math.floor(Math.random() * w.length)]); return; }
    if (b.classList.contains('bm-toggle') && state.current) { toggleBm(state.current); return; }
    if (b.classList.contains('memo-toggle') && state.current) { toggleMemo(state.current); return; }
    // 問題演習
    if (b.dataset.scope) { quizCfg.scope = b.dataset.scope; save('jd-quizcfg', quizCfg); renderQuizSetup(); return; }
    if (b.dataset.count) { quizCfg.count = Number(b.dataset.count); save('jd-quizcfg', quizCfg); renderQuizSetup(); return; }
    if (b.hasAttribute('data-timed')) { quizCfg.timed = !quizCfg.timed; save('jd-quizcfg', quizCfg); renderQuizSetup(); return; }
    if (b.hasAttribute('data-start') || b.hasAttribute('data-again')) {
      var pool = shuffle(poolFor(quizCfg));
      if (quizCfg.scope === 'weak') pool.sort(function (x, y) { return ((DB.ans[x.id] || {}).r === 0 ? 0 : 1) - ((DB.ans[y.id] || {}).r === 0 ? 0 : 1); });
      if (!pool.length) { renderQuizSetup(); return; }
      startSession('quiz', pool.slice(0, quizCfg.count), '問題演習');
      return;
    }
    if (b.hasAttribute('data-resume')) { renderSession(); return; }
    if (b.hasAttribute('data-review')) { startSession('review', dueList().slice(0, 20), '今日の復習'); return; }
    if (b.hasAttribute('data-next')) { state.session.i++; renderSession(); return; }
    if (b.hasAttribute('data-quit')) {
      var s2 = state.session;
      if (s2.res.length) { s2.list = s2.list.slice(0, s2.i + (s2.answered ? 1 : 0)); s2.i = s2.list.length; renderResult(); }
      else { state.session = null; state.mode === 'review' ? renderReview() : renderQuizSetup(); }
      return;
    }
    if (b.hasAttribute('data-retry-wrong')) {
      var wr = state.session.res.filter(function (r) { return !r.ok; }).map(function (r) { return QBY[r.id]; });
      startSession(state.session.kind, shuffle(wr), state.session.title);
      return;
    }
    if (b.hasAttribute('data-setup')) { state.session = null; return; }   // href で問題演習へ
    // 記録・設定
    if (b.dataset.font) { setFont(b.dataset.font); renderStats(); return; }
    if (b.hasAttribute('data-exam-clear')) { DB.exam = ''; persist(true); renderStats(); return; }
    // 用語カード
    if (b.hasAttribute('data-cards')) { startCards(); return; }
    if (b.hasAttribute('data-card-flip')) { var bk = detailEl.querySelector('.fc-back'); bk.hidden = false; b.hidden = true; detailEl.querySelector('.fc-judge').hidden = false; showMath(bk); return; }
    if (b.dataset.cardJudge) {
      var cs = state.cards, cid = cs.list[cs.i];
      if (b.dataset.cardJudge === '1') { cs.known++; if (!DB.memo[cid]) { DB.memo[cid] = Date.now(); persist(); } }
      else cs.again.push(cid);
      cs.i++; renderCards(); return;
    }
    if (b.hasAttribute('data-cards-again')) { var ag = state.cards.again; state.cards = { list: shuffle(ag), i: 0, known: 0, again: [] }; renderCards(); return; }
    if (b.hasAttribute('data-export')) { exportData(); return; }
    if (b.dataset.imp) {
      var box = detailEl.querySelector('.import-box'), inc = box && box.__inc;
      if (b.dataset.imp === 'cancel' || !inc) { box.hidden = true; return; }
      DB = b.dataset.imp === 'replace' ? inc : mergeDB(DB, inc);
      persist(true);
      if (/^(s|m|l)$/.test(DB.font)) document.documentElement.setAttribute('data-font', DB.font);
      renderStats(); renderList(); updateBadge();
      var nb = detailEl.querySelector('.import-box'); nb.hidden = false;
      nb.innerHTML = '<p class="msg ok">' + (b.dataset.imp === 'replace' ? '上書き' : '統合') + 'しました。</p>';
    }
  });
  detailEl.addEventListener('change', function (e) {
    var t = e.target;
    if (t.dataset.cfg) { quizCfg[t.dataset.cfg] = Number(t.value); save('jd-quizcfg', quizCfg); renderQuizSetup(); }
    if (t.hasAttribute('data-import') && t.files && t.files[0]) importFile(t.files[0]);
    if (t.hasAttribute('data-exam')) { DB.exam = /^\d{4}-\d{2}-\d{2}$/.test(t.value) ? t.value : ''; persist(true); renderStats(); }
  });

  // ---------- スマホの一覧／解説の切り替え ----------
  var listScroll = 0;
  function showDetailView() {
    if (!document.body.classList.contains('show-detail')) listScroll = listEl.scrollTop;
    document.body.classList.add('show-detail');
  }
  function showListView() {
    var was = document.body.classList.contains('show-detail');
    document.body.classList.remove('show-detail');
    if (was) listEl.scrollTop = listScroll;
  }
  function backToList() {
    if (state.fromList && history.length > 1) { state.fromList = false; history.back(); return; }
    go('#all');
  }

  // ---------- ルーティング ----------
  function go(hash) {
    if (location.hash === hash) route(); else location.hash = hash;
  }
  function scrollRowIntoView() {
    var row = listEl.querySelector('.row[aria-current="true"]');
    if (!row) return;
    var lr = listEl.getBoundingClientRect(), rr = row.getBoundingClientRect(), top = lr.top + topHeight();
    if (rr.top < top + 8 || rr.bottom > lr.bottom) listEl.scrollTop += rr.top - top - (lr.bottom - top) / 3;
  }
  // 今見ている用語の部と章に合わせる
  function focusChapter(n) {
    var t = partOf(n), changed = state.tab !== t || !state.open[n];
    state.tab = t; state.open[n] = true;
    if (changed) saveListState();
  }
  function setMode(m) {
    var was = state.mode;
    state.mode = m;
    document.body.setAttribute('data-mode', m);
    Array.prototype.forEach.call(document.querySelectorAll('.modes a'), function (a) {
      if (a.dataset.mode === m) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
      if (a.dataset.mode === 'dict') a.setAttribute('href', state.dictHash);
    });
    return was !== m;
  }
  function route() {
    var h = decodeURIComponent(location.hash.replace(/^#/, ''));
    var m = /^ch(\d{1,2})$/.exec(h);
    var fromDetail = document.body.classList.contains('show-detail');
    if (ignoreNextRoute && h === 'quiz') { ignoreNextRoute = false; return; }
    if (!h || h === 'home') {
      setMode('home'); renderHome();
      document.title = NAME;
      return;
    }
    if (h === 'cards') {
      setMode('quiz'); renderCards();
      document.title = '用語カード｜' + NAME;
      return;
    }
    var em = /^exam(?:\/(\d+)(?:\/(run|result)(?:\/(\d+))?)?)?$/.exec(h);
    if (em) {
      setMode('exam');
      var no = Number(em[1]);
      if (!em[1]) renderExamList();
      else if (em[2] === 'run') { if (DB.mock.cur && DB.mock.cur.s === no) renderExamRun(); else renderExamStart(no); }
      else if (em[2] === 'result') renderExamResult(no, em[3] ? Number(em[3]) : 0);
      else renderExamStart(no);
      document.title = '模擬試験｜' + NAME;
      return;
    }
    if (h === 'roadmap') {
      setMode('roadmap'); renderRoadmap();
      document.title = 'ロードマップ｜' + NAME;
      return;
    }
    if (h === 'quiz' || h === 'review' || h === 'stats') {
      setMode(h);
      if (h === 'stats') renderStats();
      else if (state.session && state.session.kind === h && state.session.i < state.session.list.length && state.session.res.length) renderSession();
      else h === 'quiz' ? renderQuizSetup() : renderReview();
      document.title = { quiz: '問題演習', review: '今日の復習', stats: '学習記録' }[h] + '｜' + NAME;
      return;
    }
    if (setMode('dict') && !state.current) clearDetail();
    state.dictHash = '#' + (h || 'all');
    setMode('dict');
    if (h === 'formulas') {
      renderList(); renderFormulas(); showDetailView();
      document.title = '公式まとめ｜' + NAME;
      return;
    }
    if (!h || h === 'all') {
      renderList(); showListView(); revealTab();
      if (!state.current) renderWelcome();
      else if (!detailEl.querySelector('.term-title')) renderTerm(state.current);
      return;
    }
    if (m) {
      var n = Number(m[1]);
      if (n >= 1 && n <= CHAPTERS.length) {
        if (state.collapsed) setCollapsed(false);   // 章を開く操作では一覧を見せる
        focusChapter(n);
        renderList(); showListView(); revealTab();
        if (!fromDetail && !state.query) scrollChapterIntoView(n);
        if (!state.current) renderWelcome();
        else if (!detailEl.querySelector('.term-title')) renderTerm(state.current);
        return;
      }
    }
    if (PLAN[h]) {
      var changed = state.current !== h || !detailEl.querySelector('.term-title');
      state.current = h;
      focusChapter(PLAN[h].chapter);
      renderList(); revealTab();
      if (changed) {
        if (isWritten(h)) {
          renderTerm(h);
          save('jd-last', h);
          DB.recent = [h].concat(DB.recent.filter(function (x) { return x !== h; })).slice(0, 30);
          bumpDay('v');
          persist();
        } else renderPending(h);
      }
      showDetailView();
      scrollRowIntoView();
      document.title = PLAN[h].title + '｜' + NAME;
      return;
    }
    renderList(); renderWelcome(); showListView();
  }
  window.addEventListener('hashchange', route);

  // ---------- 検索 ----------
  var timer = null, rememberTimer = null;
  function rememberSearch(s) {
    s = s.trim();
    if (!s) return;
    DB.searches = [s].concat(DB.searches.filter(function (x) { return x !== s; })).slice(0, 8);
    persist();
  }
  // PC・iPad で左の一覧をたたむ（解説を広く読むため）。状態は保存する
  var listBtn = $('btn-list');
  function setCollapsed(v) {
    state.collapsed = !!v;
    document.body.classList.toggle('list-collapsed', state.collapsed);
    listBtn.setAttribute('aria-expanded', String(!state.collapsed));
    listBtn.setAttribute('aria-label', state.collapsed ? '用語の一覧をひらく' : '用語の一覧をたたむ');
    save('jd-list-collapsed', state.collapsed);
    if (window.Charts) setTimeout(function () { Charts.redrawAll(); }, 50);
  }
  listBtn.addEventListener('click', function () {
    if (state.mode !== 'dict') { setCollapsed(false); go(state.dictHash); return; }
    setCollapsed(!state.collapsed);
    if (!state.collapsed) { revealTab(); scrollRowIntoView(); }
  });
  function runSearch() {
    var was = state.query;
    state.query = qEl.value.trim();
    if (state.query && state.collapsed) setCollapsed(false);   // 検索結果は一覧に出すので、ひらく
    if (state.mode !== 'dict') { go(state.dictHash); }
    renderList();
    if (state.query) { listEl.scrollTop = 0; showListView(); }
    else if (was) { revealTab(); listEl.scrollTop = 0; scrollRowIntoView(); }
    clearTimeout(rememberTimer);
    if (state.query) rememberTimer = setTimeout(function () { rememberSearch(state.query); }, 2500);
  }
  qEl.addEventListener('input', function () {
    clearTimeout(timer);
    timer = setTimeout(runSearch, 100);
  });
  qEl.addEventListener('focus', function () {
    state.searchFocus = true;
    if (!state.query && DB.searches.length) {
      if (state.mode !== 'dict') go(state.dictHash);
      renderList(); showListView();
    }
  });
  qEl.addEventListener('blur', function () {
    state.searchFocus = false;
    if (!state.query && DB.searches.length && state.mode === 'dict') setTimeout(function () { if (!state.searchFocus) renderList(); }, 150);
  });
  document.addEventListener('keydown', function (e) {
    var tag = (e.target.tagName || '').toLowerCase();
    var typing = tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === '/' && !typing) { e.preventDefault(); qEl.focus(); qEl.select(); }
    else if (e.key === '\\' && !typing && state.mode === 'dict') { e.preventDefault(); listBtn.click(); }
    else if (e.key === 'Escape') {
      if (qEl.value || document.activeElement === qEl) { qEl.value = ''; state.query = ''; renderList(); revealTab(); scrollRowIntoView(); qEl.blur(); }
    } else if (!typing && state.mode === 'dict' && (e.key === 'j' || e.key === 'k')) {
      var base = state.current || ORDER[0];
      var nx = state.current ? globalNeighbor(base, e.key === 'j' ? 1 : -1) : base;
      if (nx) { state.fromList = false; go('#' + nx); }
    } else if (!typing && state.mode !== 'dict' && detailEl.querySelector('.sess-card .qcard')) {
      // 問題演習中：1〜4（A〜D）で選ぶ、t/f で正誤、Enter で次へ
      var card = detailEl.querySelector('.sess-card .qcard'), k = e.key.toLowerCase(), opts = card.querySelectorAll('.q-opt');
      var idx = '1234'.indexOf(k) >= 0 ? '1234'.indexOf(k) : 'abcd'.indexOf(k);
      if (idx >= 0 && opts[idx] && !opts[idx].disabled) { e.preventDefault(); opts[idx].click(); }
      else if (e.key === 'Enter') { var nx2 = detailEl.querySelector('.sess-next:not([hidden]) [data-next]'); if (nx2) { e.preventDefault(); nx2.click(); } }
      else if (k === 'h') { var hb = card.querySelector('[data-act="hint"]'); if (hb) hb.click(); }
    } else if (!typing && e.key === 'b' && state.mode === 'dict' && state.current && isWritten(state.current) && detailEl.querySelector('.bm-toggle')) {
      toggleBm(state.current);
    }
  });

  // ---------- テーマ ----------
  var mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  function effectiveTheme() {
    var t = document.documentElement.getAttribute('data-theme');
    return t || (mq && mq.matches ? 'dark' : 'light');
  }
  $('btn-theme').addEventListener('click', function () {
    var next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('jd-theme', next); } catch (e) { /* 無視 */ }
    if (window.Charts) Charts.redrawAll();
  });
  if (mq && mq.addEventListener) mq.addEventListener('change', function () { if (window.Charts) Charts.redrawAll(); });

  // ---------- オフライン（GitHub Pages など https のときだけ） ----------
  if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost')) {
    window.addEventListener('load', function () {
      // sw.js 自体は HTTP キャッシュを使わずに確かめ、開くたびに新しい版がないか見る
      try {
        navigator.serviceWorker.register(SITE.level && SITE.level !== 'pre1' ? '../sw.js' : 'sw.js', { updateViaCache: 'none' }).then(function (reg) { try { reg.update(); } catch (e) { /* 無視 */ } }).catch(function () { /* 使えなくても続ける */ });
      } catch (e) { /* 無視 */ }
    });
  }

  // ---------- 開始 ----------
  document.documentElement.setAttribute('data-font', DB.font);
  setCollapsed(!!load('jd-list-collapsed', false));
  // 開いたときはホームから（「続きから読む」はホームにある）
  updateBadge();
  route();
})();
