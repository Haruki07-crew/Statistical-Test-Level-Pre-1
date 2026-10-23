/* 画面・検索・ルーティング・問題演習・学習記録 */
(function () {
  'use strict';

  var DATA = JSON.parse(document.getElementById('data').textContent);
  var TERMS = DATA.terms;
  var CHAPTERS = DATA.chapters;
  var QS = DATA.questions || [];
  var FORMULAS = DATA.formulas || '';
  var PARTS = [
    ['確率と分布', 1, 6], ['漸近理論と推測', 7, 13], ['確率過程', 14, 15], ['回帰分析', 16, 19],
    ['実験計画と標本調査', 20, 21], ['多変量解析', 22, 26], ['時系列・分割表・不完全データ', 27, 29],
    ['モデル選択・ベイズ・シミュレーション', 30, 32], ['統計的因果推論', 33, 33]
  ];
  var SEC_LABELS = { summary: '一言', definition: '定義', reading: '読み解き', example: '具体例', field: '現場', chart: 'グラフ', exam: '試験', mistake: '間違い', questions: '問題', note: 'メモ' };
  var TYPE_LABELS = { choice: '4択', numeric: '数値', truefalse: '正誤', derivation: '導出' };
  var DAY = 86400000;
  var INTERVALS = [1, 3, 7, 14, 30];   // 正解が続いた回数ごとの、次に出すまでの日数

  var $ = function (id) { return document.getElementById(id); };
  var listEl = $('list'), detailEl = $('detail'), qEl = $('q');

  // ---------- 小さな保存（タブ・開閉など。使えなくても動くように） ----------
  function load(key, def) { try { var v = localStorage.getItem(key); return v == null ? def : JSON.parse(v); } catch (e) { return def; } }
  function save(key, val) { try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) { /* 保存できなくても続ける */ } }

  // ---------- 学習データ（版番号つき。壊れていても落ちない） ----------
  var DB_KEY = 'jd-data', DB_VERSION = 1;
  function isObj(x) { return x && typeof x === 'object' && !Array.isArray(x); }
  function num(x, def) { return typeof x === 'number' && isFinite(x) ? x : def; }
  function emptyDB() { return { v: DB_VERSION, memo: {}, bm: {}, notes: {}, recent: [], ans: {}, days: {}, searches: [], font: 'm', exam: '', qbm: {} }; }
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
      PLAN[t.id] = { id: t.id, title: t.title, importance: t.importance, status: t.status, chapter: ch.no, order: i };
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
  Object.keys(TERMS).forEach(function (id) { TERMS[id].text = plainFromHtml(TERMS[id].html); });

  var INDEX = ORDER.map(function (id) {
    var p = PLAN[id], t = TERMS[id];
    return {
      id: id,
      names: [p.title].concat(t ? [t.title].concat(t.aliases) : []).map(norm),
      reading: t ? norm(t.reading) : '',
      en: t ? norm(t.en) : '',
      tags: t ? t.tags.map(norm) : [],
      text: t ? norm(t.text) : ''
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

  // ---------- 数式とグラフの遅延描画（見える部分だけ） ----------
  var io = window.IntersectionObserver ? new IntersectionObserver(function (ents) {
    ents.forEach(function (en) {
      if (!en.isIntersecting) return;
      io.unobserve(en.target);
      activate(en.target);
    });
  }, { root: null, rootMargin: '600px 0px' }) : null;
  function activate(el) {
    if (el.__spec) {
      try { Charts.render(el, el.__spec); } catch (e) { if (window.console) console.error(e); }
      el.classList.remove('chart-lazy');
      return;
    }
    typeset(el);
  }
  function lazy(el) { if (io) io.observe(el); else activate(el); }
  var mjReady = false, waitTypeset = [];
  function typeset(el) {
    var MJ = window.MathJax;
    if (!el) return;
    if (!mjReady || !MJ || !MJ.typesetPromise) { if (waitTypeset.indexOf(el) < 0) waitTypeset.push(el); return; }
    MJ.typesetPromise([el]).catch(function (err) { if (window.console) console.warn(err); });
  }
  function mathReady() {
    if (mjReady || !window.MathJax || !window.MathJax.typesetPromise) return;
    mjReady = true;
    var els = waitTypeset.filter(function (el) { return el.isConnected; });
    waitTypeset = [];
    els.forEach(typeset);
  }
  document.addEventListener('mathjax-ready', mathReady);
  // 準備完了の合図が届かない場合（オフラインで一部の読み込みに失敗したときなど）にも描けるようにする
  window.addEventListener('load', function () { setTimeout(mathReady, 1500); });
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
  function renderWelcome() {
    clearDetail();
    var tot = ORDER.length, wr = ORDER.filter(isWritten).length, vf = ORDER.filter(function (id) { return PLAN[id].status === 'verified'; }).length;
    var last = DB.recent.filter(isWritten)[0];
    var due = dueList().length, st = streakDays();
    var h = '<div class="article welcome">' +
      '<h1>統計検定準1級 用語辞典</h1>' +
      '<div class="home-cards">' +
      (last ? '<button type="button" class="home-card" data-go="' + last + '"><small>続きから読む</small><b>' + esc(PLAN[last].title) + '</b></button>' : '') +
      '<a class="home-card" href="#review"><small>今日の復習</small><b>' + due + ' 問</b></a>' +
      '<a class="home-card" href="#quiz"><small>問題演習</small><b>' + QS.length + ' 問から出題</b></a>' +
      '<a class="home-card" href="#stats"><small>連続学習</small><b>' + st + ' 日</b></a>' +
      (examDays() !== null ? '<a class="home-card" href="#stats"><small>試験まで</small><b>' + (examDays() >= 0 ? 'あと ' + examDays() + ' 日' : '終了') + '</b></a>' : '') +
      '<a class="home-card" href="#cards"><small>用語カード</small><b>めくって覚える</b></a>' +
      '<button type="button" class="home-card" data-random><small>ランダムに1語</small><b>どれか開く</b></button>' +
      '<a class="home-card" href="#formulas"><small>公式まとめ</small><b>分布・検定・推定</b></a>' +
      '</div>';
    // 今日の1問：日付で決まる1問（復習待ちがあればその先頭）
    var dq = dueList()[0];
    if (!dq && QS.length) { var seed = Number(todayKey().replace(/-/g, '')); dq = QS[(seed * 2654435761 % 4294967296) % QS.length]; }
    if (dq) h += '<h2 class="home-h">今日の1問</h2><div class="home-q"></div>';
    var recent = DB.recent.filter(isWritten).slice(0, 8);
    if (recent.length) h += '<h2 class="home-h">最近見た用語</h2><div class="related-list">' + recent.map(function (id) { return termBtn(id); }).join('') + '</div>';
    var bms = Object.keys(DB.bm).filter(isWritten).sort(function (a, b) { return DB.bm[b] - DB.bm[a]; });
    if (bms.length) h += '<h2 class="home-h">ブックマーク</h2><div class="related-list">' + bms.slice(0, 30).map(function (id) { return termBtn(id); }).join('') + '</div>';
    h += '<p class="home-note">『統計学実践ワークブック』（増補版）の全33章の章立てに沿って、用語の定義・数式・具体例・試験のポイントをまとめています。解説と問題はすべてオリジナルです。' +
      '作成済み ' + wr + ' 語 ／ 予定 ' + tot + ' 語（うち数値検算済み ' + vf + ' 語）、検算済みの問題 ' + QS.length + ' 問。</p>' +
      '<h2 class="home-h">使い方</h2>' +
      '<ul class="home-list"><li>一覧の上のタブで部を選び、章の見出しを押すと用語が開きます。上の検索欄に用語・読み・英語名を入れると全章から探せます（「ガウス」「がんま」「MGF」など）。</li>' +
      '<li>用語ページで「覚えた」チェック・★ブックマーク・自分のメモを付けられます。ページの最後にその用語の問題があります。</li>' +
      '<li>「問題」で範囲と問題数を選んで解き、「復習」で間違えた問題や時間が経った問題を解き直します（正解が続くほど 1→3→7→14→30 日と間隔が空きます）。</li>' +
      '<li>PCでは <kbd>/</kbd> で検索、<kbd>j</kbd>／<kbd>k</kbd> で次・前の用語、<kbd>b</kbd> でブックマーク、<kbd>Esc</kbd> で検索の解除。問題演習では <kbd>1</kbd>〜<kbd>4</kbd> で選択、<kbd>Enter</kbd> で次へ、<kbd>h</kbd> でヒント。</li>' +
      '<li>「記録」で文字サイズの変更と、学習データのファイルへの書き出し・読み込み（iPad とスマホの間で移す）ができます。</li></ul>' +
      '<h2 class="home-h">重要度の印</h2>' +
      '<div class="legend"><span>' + impMark('A') + '必須・頻出</span><span>' + impMark('B') + '重要</span><span>' + impMark('C') + '補足</span></div>' +
      '</div>';
    detailEl.innerHTML = h;
    var hq = detailEl.querySelector('.home-q');
    if (hq) hq.appendChild(qCard(dq, { termLink: true, retry: true }));
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
    Array.prototype.forEach.call(detailEl.querySelectorAll('.term-title, .term-meta'), typeset);
    Array.prototype.forEach.call(detailEl.querySelectorAll('.sec'), lazy);
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
      '<div class="q-exp" hidden><b>解説</b>' + q.e + '</div>' +
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
      typeset(el);
      if (opts.onAnswer) opts.onAnswer(ok, el);
    }
    el.addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (!b || b.disabled) return;
      var act = b.dataset.act;
      if (act === 'hint') {
        var hh = el.querySelector('.q-hint'); hh.hidden = !hh.hidden; b.setAttribute('aria-expanded', String(!hh.hidden));
        if (!hh.hidden) typeset(hh);
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
        typeset(li);
        if (stepI >= q.steps.length) {
          b.hidden = true;
          var fin = el.querySelector('.q-final'); fin.hidden = false; typeset(fin);
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
    lazy(el);
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
  if (!isObj(quizCfg)) quizCfg = { scope: 'all', ch: 1, part: 0, count: 10 };

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
    state.session = { kind: kind, list: list.map(function (q) { return q.id; }), i: 0, res: [], title: title, answered: false };
    renderSession();
  }
  function renderSession() {
    clearDetail();
    var s = state.session;
    if (!s) { state.mode === 'review' ? renderReview() : renderQuizSetup(); return; }
    if (s.i >= s.list.length) { renderResult(); return; }
    var q = QBY[s.list[s.i]];
    s.answered = false;
    detailEl.innerHTML = '<div class="article screen"><div class="sess-head"><button type="button" class="q-btn ghost" data-quit>やめる</button><b>' + esc(s.title) + '</b><span class="sess-n">' + (s.i + 1) + ' / ' + s.list.length + '</span></div>' +
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
  function renderResult() {
    var s = state.session, ok = s.res.filter(function (r) { return r.ok; }).length, n = s.res.length;
    var wrong = s.res.filter(function (r) { return !r.ok; }).map(function (r) { return QBY[r.id]; });
    var terms = {};
    wrong.forEach(function (q) { terms[q.term] = true; });
    var h = '<div class="article screen"><h1 class="screen-h">' + esc(s.title) + 'の結果</h1>' +
      '<div class="score"><b>' + (n ? Math.round(100 * ok / n) : 0) + '%</b><span>' + ok + ' / ' + n + ' 問正解</span></div>';
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
    Array.prototype.forEach.call(detailEl.querySelectorAll('.lazy-math'), lazy);
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
      '<div class="fc-back body" hidden>' + summaryHtml(id) + '<p>' + termBtn(id) + '</p></div>' +
      '<button type="button" class="big-btn" data-card-flip>めくる</button>' +
      '<div class="fc-judge q-row" hidden><button type="button" class="q-btn primary" data-card-judge="1">覚えた</button><button type="button" class="q-btn" data-card-judge="0">まだ</button></div></div></div>';
    detailEl.scrollTop = 0;
    typeset(detailEl.querySelector('.fc-title'));
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
      box.innerHTML = '<p>読み込むデータ：覚えた ' + Object.keys(inc.memo).length + ' 語、ブックマーク ' + Object.keys(inc.bm).length + ' 語、メモ ' + Object.keys(inc.notes).length + ' 件、回答した問題 ' + Object.keys(inc.ans).length + ' 問（' + esc(String(d.exported || '').slice(0, 10)) + ' に書き出し）</p>' +
        '<div class="seg-row"><button type="button" class="chip" data-imp="merge">今の記録と統合する</button><button type="button" class="chip" data-imp="replace">今の記録を上書きする</button><button type="button" class="chip" data-imp="cancel">やめる</button></div>';
      box.__inc = inc;
    };
    rd.readAsText(file);
  }

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
    Array.prototype.forEach.call(detailEl.querySelectorAll('.sec'), lazy);
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
    if (b.hasAttribute('data-random')) { var w = ORDER.filter(isWritten); go('#' + w[Math.floor(Math.random() * w.length)]); return; }
    if (b.classList.contains('bm-toggle') && state.current) { toggleBm(state.current); return; }
    if (b.classList.contains('memo-toggle') && state.current) { toggleMemo(state.current); return; }
    // 問題演習
    if (b.dataset.scope) { quizCfg.scope = b.dataset.scope; save('jd-quizcfg', quizCfg); renderQuizSetup(); return; }
    if (b.dataset.count) { quizCfg.count = Number(b.dataset.count); save('jd-quizcfg', quizCfg); renderQuizSetup(); return; }
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
    if (b.hasAttribute('data-card-flip')) { var bk = detailEl.querySelector('.fc-back'); bk.hidden = false; b.hidden = true; detailEl.querySelector('.fc-judge').hidden = false; typeset(bk); return; }
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
    if (h === 'cards') {
      setMode('quiz'); renderCards();
      document.title = '用語カード｜準1級 用語辞典';
      return;
    }
    if (h === 'quiz' || h === 'review' || h === 'stats') {
      setMode(h);
      if (h === 'stats') renderStats();
      else if (state.session && state.session.kind === h && state.session.i < state.session.list.length && state.session.res.length) renderSession();
      else h === 'quiz' ? renderQuizSetup() : renderReview();
      document.title = { quiz: '問題演習', review: '今日の復習', stats: '学習記録' }[h] + '｜準1級 用語辞典';
      return;
    }
    if (setMode('dict') && !state.current) clearDetail();
    state.dictHash = '#' + (h || 'all');
    setMode('dict');
    if (h === 'formulas') {
      renderList(); renderFormulas(); showDetailView();
      document.title = '公式まとめ｜準1級 用語辞典';
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
      document.title = PLAN[h].title + '｜準1級 用語辞典';
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
  function runSearch() {
    var was = state.query;
    state.query = qEl.value.trim();
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
      try { navigator.serviceWorker.register('sw.js').catch(function () { /* 使えなくても続ける */ }); } catch (e) { /* 無視 */ }
    });
  }

  // ---------- 開始 ----------
  document.documentElement.setAttribute('data-font', DB.font);
  if (!location.hash) {
    var last = DB.recent.filter(isWritten)[0];
    if (last && window.innerWidth >= 600) {
      state.current = last;
      history.replaceState(null, '', '#' + last);
    }
  }
  updateBadge();
  route();
})();
