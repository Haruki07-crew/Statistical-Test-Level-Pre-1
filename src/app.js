/* 画面・検索・ルーティング */
(function () {
  'use strict';

  var DATA = JSON.parse(document.getElementById('data').textContent);
  var TERMS = DATA.terms;
  var CHAPTERS = DATA.chapters;
  var PARTS = [
    ['確率と分布', 1, 6], ['漸近理論と推測', 7, 13], ['確率過程', 14, 15], ['回帰分析', 16, 19],
    ['実験計画と標本調査', 20, 21], ['多変量解析', 22, 26], ['時系列・分割表・不完全データ', 27, 29],
    ['モデル選択・ベイズ・シミュレーション', 30, 32]
  ];
  var SEC_LABELS = { summary: '一言', definition: '定義', reading: '読み解き', example: '具体例', field: '現場', chart: 'グラフ', exam: '試験', mistake: '間違い' };

  var $ = function (id) { return document.getElementById(id); };
  var navEl = $('nav'), listEl = $('list'), detailEl = $('detail'), qEl = $('q'), scrim = $('scrim');

  // ---------- 保存（使えなくても動くように） ----------
  function load(key, def) { try { var v = localStorage.getItem(key); return v == null ? def : JSON.parse(v); } catch (e) { return def; } }
  function save(key, val) { try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) { /* 保存できなくても続ける */ } }

  var memo = {};
  (load('jd-memo', []) || []).forEach(function (id) { memo[id] = true; });
  function saveMemo() { save('jd-memo', Object.keys(memo).filter(function (k) { return memo[k]; })); }

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
  function isWritten(id) { return !!TERMS[id]; }

  // 検索用の正規化：NFKC → 小文字 → カタカナをひらがな → 空白・中黒・ハイフン類を除去
  function norm(s) {
    s = String(s || '');
    try { s = s.normalize('NFKC'); } catch (e) { /* 古い環境 */ }
    s = s.toLowerCase().replace(/[ァ-ヶ]/g, function (c) { return String.fromCharCode(c.charCodeAt(0) - 0x60); });
    return s.replace(/[\s　・･\-‐‑‒–—―−]/g, '');
  }
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
    chapter: 'all',         // 'all' か章番号
    query: '',
    hideMemo: !!load('jd-hide-memo', false),
    current: null,
    fromList: false
  };

  // ---------- 表示の部品 ----------
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function impMark(imp) {
    var title = { A: '重要度A（必須・頻出）', B: '重要度B（重要）', C: '重要度C（補足）' }[imp] || '';
    return '<span class="imp imp-' + imp + '" title="' + title + '" aria-label="' + title + '">' + imp + '</span>';
  }
  var CHECK_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M7.5 12.5l3 3 6-6.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="tick"/></svg>';
  var CHECK_OFF = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>';

  function chapterCounts(ch) {
    var w = 0;
    ch.terms.forEach(function (t) { if (isWritten(t.id)) w++; });
    return { written: w, total: ch.terms.length };
  }

  // ---------- 章ナビ ----------
  function renderNav() {
    var h = '<div class="nav-head">' +
      '<button type="button" data-nav="all" aria-pressed="' + (state.chapter === 'all') + '">すべての章</button>' +
      '<button type="button" data-nav="hide-memo" aria-pressed="' + state.hideMemo + '">覚えた用語を隠す</button></div>';
    PARTS.forEach(function (p) {
      var open = state.chapter === 'all' || (state.chapter >= p[1] && state.chapter <= p[2]);
      h += '<details' + (open ? ' open' : '') + '><summary>' + esc(p[0]) + '</summary>';
      for (var n = p[1]; n <= p[2]; n++) {
        var ch = chapterOf(n), c = chapterCounts(ch);
        h += '<button type="button" class="ch-btn" data-ch="' + n + '"' + (state.chapter === n ? ' aria-current="true"' : '') + '>' +
          '<span class="ch-no">' + String(n).padStart(2, '0') + '</span><span>' + esc(ch.title) + '</span>' +
          '<span class="ch-count" title="作成済み／予定">' + c.written + '/' + c.total + '</span></button>';
      }
      h += '</details>';
    });
    navEl.innerHTML = h;
  }

  navEl.addEventListener('click', function (e) {
    var b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.nav === 'all') { go('#all'); closeNav(); }
    else if (b.dataset.nav === 'hide-memo') {
      state.hideMemo = !state.hideMemo; save('jd-hide-memo', state.hideMemo);
      renderNav(); renderList();
    } else if (b.dataset.ch) { go('#ch' + String(b.dataset.ch).padStart(2, '0')); closeNav(); }
  });

  function openNav() { document.body.classList.add('nav-open'); scrim.hidden = false; $('btn-nav').setAttribute('aria-expanded', 'true'); }
  function closeNav() { document.body.classList.remove('nav-open'); scrim.hidden = true; $('btn-nav').setAttribute('aria-expanded', 'false'); }
  $('btn-nav').addEventListener('click', function () { document.body.classList.contains('nav-open') ? closeNav() : openNav(); });
  scrim.addEventListener('click', closeNav);

  // ---------- 用語一覧 ----------
  function rowHtml(id, opts) {
    var p = PLAN[id], t = TERMS[id], written = !!t;
    var cur = state.current === id ? ' aria-current="true"' : '';
    var title = esc(p.title);
    var sub = written ? esc(t.reading) + (t.en ? '　' + esc(t.en) : '') : '';
    var snip = '';
    if (opts && opts.snip) snip = '<span class="row-snippet">' + esc(opts.snip.pre) + '<mark>' + esc(opts.snip.hit) + '</mark>' + esc(opts.snip.post) + '</span>';
    var chTag = opts && opts.showCh ? '<span class="row-ch">第' + p.chapter + '章</span>' : '';
    var memoBtn = written
      ? '<button type="button" class="memo-btn" data-memo="' + id + '" aria-pressed="' + !!memo[id] + '" aria-label="覚えた">' + (memo[id] ? CHECK_SVG : CHECK_OFF) + '</button>'
      : '';
    return '<li class="row' + (written ? '' : ' pending') + '" data-id="' + id + '"' + cur + '>' +
      '<button type="button" class="row-main"' + (written ? '' : ' aria-disabled="true" tabindex="-1"') + '>' +
      '<span class="row-title"><span>' + title + '</span>' + impMark(p.importance) + (written ? '' : '<span class="pending-tag">準備中</span>') + chTag + '</span>' +
      (sub ? '<span class="row-reading">' + sub + '</span>' : '') + snip + '</button>' + memoBtn + '</li>';
  }

  function inScope(id) {
    if (state.chapter !== 'all' && PLAN[id].chapter !== state.chapter) return false;
    if (state.hideMemo && memo[id]) return false;
    return true;
  }

  function renderList() {
    var h = '', q = norm(state.query);
    var scopeName = state.chapter === 'all' ? '全章' : '第' + state.chapter + '章 ' + chapterOf(state.chapter).title;
    if (q) {
      var res = [];
      INDEX.forEach(function (e) {
        if (!inScope(e.id)) return;
        var s = scoreEntry(e, q);
        if (s) res.push({ id: e.id, s: s });
      });
      res.sort(function (a, b) { return b.s - a.s || ORDER.indexOf(a.id) - ORDER.indexOf(b.id); });
      h += '<div class="list-head"><h2 class="list-title">「' + esc(state.query) + '」の検索結果 ' + res.length + '件</h2><p class="list-sub">' + esc(scopeName) + 'から</p></div>';
      if (res.length) {
        h += '<ul>' + res.map(function (r) {
          return rowHtml(r.id, { showCh: true, snip: r.s === 10 ? snippet(r.id, q) : null });
        }).join('') + '</ul>';
      } else {
        var near = nearCandidates(q);
        h += '<div class="empty"><h3>見つかりませんでした</h3>' + (state.chapter !== 'all' ? '<p>章の絞り込みを外すと見つかるかもしれません。</p>' : '') +
          (near.length ? '<p>近い候補：</p></div><ul>' + near.map(function (id) { return rowHtml(id, { showCh: true }); }).join('') + '</ul>' : '</div>');
      }
    } else if (state.chapter === 'all') {
      var tot = ORDER.length, wr = ORDER.filter(isWritten).length;
      h += '<div class="list-head"><h2 class="list-title">すべての用語</h2><p class="list-sub">作成済み ' + wr + ' ／ 予定 ' + tot + '</p></div>';
      CHAPTERS.forEach(function (ch) {
        var ids = ch.terms.map(function (t) { return t.id; }).filter(inScope);
        if (!ids.length) return;
        h += '<div class="list-head" style="position:static;padding-top:14px"><h3 class="list-title" style="font-size:.85rem">第' + ch.no + '章 ' + esc(ch.title) + '</h3></div>';
        h += '<ul>' + ids.map(function (id) { return rowHtml(id); }).join('') + '</ul>';
      });
    } else {
      var ch = chapterOf(state.chapter), c = chapterCounts(ch);
      var ids = ch.terms.map(function (t) { return t.id; }).filter(inScope);
      h += '<div class="list-head"><h2 class="list-title">第' + ch.no + '章 ' + esc(ch.title) + '</h2><p class="list-sub">作成済み ' + c.written + ' ／ 予定 ' + c.total + (state.hideMemo ? '（覚えた用語を隠しています）' : '') + '</p></div>';
      h += ids.length ? '<ul>' + ids.map(function (id) { return rowHtml(id); }).join('') + '</ul>' : '<div class="empty">表示する用語がありません。</div>';
    }
    listEl.innerHTML = h;
  }

  listEl.addEventListener('click', function (e) {
    var m = e.target.closest('[data-memo]');
    if (m) { toggleMemo(m.dataset.memo); return; }
    var row = e.target.closest('.row');
    if (!row || row.classList.contains('pending')) return;
    state.fromList = true;
    go('#' + row.dataset.id);
  });

  function toggleMemo(id) {
    memo[id] = !memo[id];
    if (!memo[id]) delete memo[id];
    saveMemo();
    renderList();
    var mt = detailEl.querySelector('.memo-toggle');
    if (mt && state.current === id) setMemoToggle(mt, id);
  }
  function setMemoToggle(btn, id) {
    btn.setAttribute('aria-pressed', String(!!memo[id]));
    btn.innerHTML = (memo[id] ? CHECK_SVG : CHECK_OFF) + (memo[id] ? '覚えた' : '覚えたらチェック');
  }

  // ---------- 解説 ----------
  function typeset(el) {
    var MJ = window.MathJax;
    if (MJ && MJ.typesetPromise) {
      MJ.typesetPromise([el]).catch(function (err) { if (window.console) console.warn(err); });
    }
  }
  document.addEventListener('mathjax-ready', function () { typeset(detailEl); });
  window.addEventListener('load', function () { if (window.Charts) Charts.redrawAll(); });

  function clearDetail() {
    if (window.Charts) Charts.destroyAll();
    var MJ = window.MathJax;
    if (MJ && MJ.typesetClear) { try { MJ.typesetClear([detailEl]); } catch (e) { /* 無視 */ } }
  }

  function renderWelcome() {
    clearDetail();
    var tot = ORDER.length, wr = ORDER.filter(isWritten).length, vf = ORDER.filter(function (id) { return PLAN[id].status === 'verified'; }).length;
    detailEl.innerHTML = '<div class="article welcome">' +
      '<h1>統計検定準1級 用語辞典</h1>' +
      '<p>『統計学実践ワークブック』の全32章の章立てに沿って、用語の定義・数式・具体例・試験のポイントをまとめています。解説はすべてオリジナルです。</p>' +
      '<div class="progress-bar" aria-hidden="true"><span style="width:' + (100 * wr / Math.max(1, tot)).toFixed(1) + '%"></span></div>' +
      '<p style="margin-top:0;font-size:.85rem">作成済み ' + wr + ' 語 ／ 予定 ' + tot + ' 語（うち数値検算済み ' + vf + ' 語）</p>' +
      '<h2 style="font-size:1rem;margin-top:1.6em">使い方</h2>' +
      '<ul style="padding-left:1.2em"><li>左の章から選ぶか、上の検索欄に用語・読み・英語名を入れます（「ガウス」「がんま」「MGF」など）。</li>' +
      '<li>各用語の右の丸をチェックすると「覚えた」印が付き、章ナビの「覚えた用語を隠す」で一覧から消せます。</li>' +
      '<li>PCでは <kbd>/</kbd> で検索、<kbd>Esc</kbd> で検索の解除ができます。</li>' +
      '<li>用語ごとのURL（#用語のid）をブックマークすると、その用語を直接開けます。</li></ul>' +
      '<h2 style="font-size:1rem;margin-top:1.6em">重要度の印</h2>' +
      '<div class="legend"><span>' + impMark('A') + '必須・頻出</span><span>' + impMark('B') + '重要</span><span>' + impMark('C') + '補足</span><span><span class="pending-tag">準備中</span>まだ書かれていない用語</span></div>' +
      '</div>';
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

  function renderTerm(id) {
    clearDetail();
    var t = TERMS[id], p = PLAN[id], ch = chapterOf(t.chapter);
    var meta = [esc(t.reading)];
    if (t.en) meta.push('<span class="en" lang="en">' + esc(t.en) + '</span>');
    if (t.aliases.length) meta.push('別名：' + t.aliases.map(esc).join('、'));
    var h = '<button type="button" class="back-btn" data-back>← 一覧</button><div class="article">' +
      '<p class="crumb">第' + t.chapter + '章 <button type="button" data-goch="' + t.chapter + '">' + esc(ch.title) + '</button></p>' +
      '<h1 class="term-title">' + esc(t.title) + impMark(t.importance) + '</h1>' +
      '<p class="term-meta">' + meta.join('　・　') + '</p>' +
      '<div class="term-actions"><button type="button" class="memo-toggle"></button>' +
      (p.status === 'verified' ? '<span class="pending-tag" style="border-style:solid" title="数値をPythonで検算済み">検算済み</span>' : '') + '</div>';
    var secs = [];
    t.html.replace(/data-sec="(\w+)"/g, function (_, k) { secs.push(k); return _; });
    if (t.charts.length && secs.indexOf('chart') < 0) secs.splice(Math.max(0, secs.indexOf('exam')), 0, 'chart');
    h += '<nav class="jump" aria-label="節へ移動">' + secs.filter(function (k) { return SEC_LABELS[k]; }).map(function (k) {
      return '<button type="button" data-jump="' + k + '">' + SEC_LABELS[k] + '</button>';
    }).join('') + '</nav>';
    h += '<div class="body">' + t.html + '</div>';
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
    setMemoToggle(detailEl.querySelector('.memo-toggle'), id);

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
        try { Charts.render(sec, spec); } catch (e) { if (window.console) console.error(e); }
      });
    }
    detailEl.scrollTop = 0;
    typeset(detailEl);
  }

  detailEl.addEventListener('click', function (e) {
    var b = e.target.closest('button');
    if (!b) return;
    if (b.hasAttribute('data-back')) { backToList(); return; }
    if (b.dataset.goch) { go('#ch' + String(b.dataset.goch).padStart(2, '0')); return; }
    if (b.dataset.go) { state.fromList = false; go('#' + b.dataset.go); return; }
    if (b.dataset.jump) {
      var s = detailEl.querySelector('.sec[data-sec="' + b.dataset.jump + '"]');
      if (s) s.scrollIntoView({ block: 'start', behavior: 'smooth' });
      return;
    }
    if (b.classList.contains('memo-toggle') && state.current) toggleMemo(state.current);
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
    go(state.chapter === 'all' ? '#all' : '#ch' + String(state.chapter).padStart(2, '0'));
  }

  // ---------- ルーティング ----------
  function go(hash) {
    if (location.hash === hash) route(); else location.hash = hash;
  }
  function scrollRowIntoView() {
    var row = listEl.querySelector('.row[aria-current="true"]');
    if (!row) return;
    var lr = listEl.getBoundingClientRect(), rr = row.getBoundingClientRect();
    if (rr.top < lr.top + 60 || rr.bottom > lr.bottom) listEl.scrollTop += rr.top - lr.top - lr.height / 3;
  }
  function route() {
    var h = decodeURIComponent(location.hash.replace(/^#/, ''));
    var m = /^ch(\d{1,2})$/.exec(h);
    if (!h || h === 'all') {
      state.chapter = 'all';
      renderNav(); renderList(); showListView();
      if (!state.current) renderWelcome();
      return;
    }
    if (m) {
      var n = Number(m[1]);
      if (n >= 1 && n <= 32) {
        if (state.chapter !== n) listEl.scrollTop = 0;
        state.chapter = n;
        renderNav(); renderList(); showListView();
        if (!state.current) renderWelcome();
        return;
      }
    }
    if (PLAN[h]) {
      state.current = h;
      if (!state.query && state.chapter !== 'all' && state.chapter !== PLAN[h].chapter) { state.chapter = PLAN[h].chapter; listEl.scrollTop = 0; }
      renderNav(); renderList();
      if (isWritten(h)) { renderTerm(h); save('jd-last', h); } else renderPending(h);
      showDetailView();
      scrollRowIntoView();
      document.title = PLAN[h].title + '｜準1級 用語辞典';
      return;
    }
    renderNav(); renderList(); renderWelcome(); showListView();
  }
  window.addEventListener('hashchange', route);

  // ---------- 検索 ----------
  var timer = null;
  qEl.addEventListener('input', function () {
    clearTimeout(timer);
    timer = setTimeout(function () {
      state.query = qEl.value.trim();
      renderList();
      listEl.scrollTop = 0;
      if (state.query) showListView();
    }, 100);
  });
  document.addEventListener('keydown', function (e) {
    var tag = (e.target.tagName || '').toLowerCase();
    var typing = tag === 'input' || tag === 'textarea' || e.target.isContentEditable;
    if (e.key === '/' && !typing && !e.metaKey && !e.ctrlKey) { e.preventDefault(); qEl.focus(); qEl.select(); }
    else if (e.key === 'Escape') {
      if (document.body.classList.contains('nav-open')) { closeNav(); return; }
      if (qEl.value || document.activeElement === qEl) { qEl.value = ''; state.query = ''; renderList(); qEl.blur(); }
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

  // ---------- 開始 ----------
  if (!location.hash) {
    var last = load('jd-last', null);
    if (last && isWritten(last) && window.innerWidth >= 600) {
      state.current = last;
      history.replaceState(null, '', '#' + last);
    }
  }
  route();
})();
