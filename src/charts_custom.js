/* カスタム図（charts.js の registerCustom で登録）。パラメータの一覧は src/CUSTOM_CHARTS.md */
(function (root) {
  'use strict';
  var Charts = root.Charts, Dist = root.Dist;
  // 乱数を使う図には「もう一度」ボタンを付ける
  var RANDOM = ['clt-sim', 'lln-path', 'ci-coverage', 'poisson-process', 'random-walk', 'regression-residual', 'qq-plot', 'bias-variance',
    'pca-axes', 'dendrogram', 'residual-pattern', 'dw-sim', 'iv-scatter', 'discriminant-boundary', 'fisher-projection', 'knn-boundary', 'svm-margin', 'kmeans-steps', 'acf-plot', 'ar-sim', 'km-curve', 'mcmc-trace', 'inverse-transform', 'rejection-sampling', 'bootstrap-hist'];
  function reg(id, fn) { Charts.registerCustom(id, fn, RANDOM.indexOf(id) >= 0); }

  // ---------- 小さな道具 ----------
  function num(v, d) { return v == null || v === '' || isNaN(Number(v)) ? d : Number(v); }
  function pts(xs, ys) { return xs.map(function (x, i) { return { x: x, y: ys[i] }; }); }
  function barDs(label, data, c, extra) {
    return Object.assign({ type: 'bar', label: label, data: data, backgroundColor: c.fill, borderColor: c.c1, borderWidth: 1, barPercentage: 1, categoryPercentage: 1 }, extra || {});
  }
  function opts(a, c, o) { return a.baseOptions(c, o); }
  function mean(v) { var s = 0; v.forEach(function (x) { s += x; }); return s / v.length; }
  function sd(v) { var m = mean(v), s = 0; v.forEach(function (x) { s += (x - m) * (x - m); }); return Math.sqrt(s / (v.length - 1)); }
  function quantileSorted(s, p) { var h = (s.length - 1) * p, lo = Math.floor(h); return s[lo] + (h - lo) * ((s[Math.min(lo + 1, s.length - 1)]) - s[lo]); }
  function sorted(v) { return v.slice().sort(function (x, y) { return x - y; }); }
  function poissonDraw(rand, lam) { var L = Math.exp(-lam), k = 0, p = 1; do { k++; p *= rand(); } while (p > L); return k - 1; }
  function expDraw(rand, rate) { return -Math.log(1 - rand()) / rate; }
  // 連立一次方程式 A x = b（ガウスの消去法、部分ピボット）
  function solve(A, b) {
    var n = b.length, M = A.map(function (r, i) { return r.slice().concat([b[i]]); }), i, j, k;
    for (i = 0; i < n; i++) {
      var piv = i;
      for (k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[piv][i])) piv = k;
      var t = M[i]; M[i] = M[piv]; M[piv] = t;
      for (k = i + 1; k < n; k++) { var f = M[k][i] / M[i][i]; for (j = i; j <= n; j++) M[k][j] -= f * M[i][j]; }
    }
    var x = new Array(n);
    for (i = n - 1; i >= 0; i--) { var s = M[i][n]; for (j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
    return x;
  }
  function lsq(X, y, ridge) {
    var p = X[0].length, A = [], b = [], i, j, k;
    for (i = 0; i < p; i++) { A.push(new Array(p).fill(0)); b.push(0); }
    for (k = 0; k < X.length; k++) for (i = 0; i < p; i++) { b[i] += X[k][i] * y[k]; for (j = 0; j < p; j++) A[i][j] += X[k][i] * X[k][j]; }
    for (i = 0; i < p; i++) A[i][i] += ridge || 0;
    return solve(A, b);
  }
  function drawArrow(ctx, x1, y1, x2, y2, head) {
    var ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(ang - 0.4), y2 - head * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - head * Math.cos(ang + 0.4), y2 - head * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
  }
  // 分布名と平らな params から、その分布のパラメータだけを取り出す
  function distParams(name, p) { var q = {}; Dist.params(name).forEach(function (k) { if (p[k] != null) q[k] = p[k]; }); return q; }

  // ---------- 汎用：データを直接渡す棒グラフ・折れ線 ----------
  // bar-data: { labels: [...], values: [...], ylabel, cumulative: true で累積割合の折れ線を重ねる }
  reg('bar-data', function (a, p, c) {
    var vals = (p.values || []).map(Number), labels = p.labels || vals.map(function (_, i) { return String(i + 1); });
    var ds = [barDs(p.label || '値', vals, c, { barPercentage: 0.8, categoryPercentage: 0.9 })];
    if (p.cumulative) {
      var tot = vals.reduce(function (s, v) { return s + v; }, 0), acc = 0;
      ds.push(a.lineDs('累積割合', vals.map(function (v) { acc += v; return acc / tot; }), c.c2, { yAxisID: 'y2', pointRadius: 3 }));
    }
    var o = opts(a, c, { xType: 'category', legend: ds.length > 1, yTitle: p.ylabel, xTitle: p.xlabel });
    if (p.cumulative) o.scales.y2 = { position: 'right', min: 0, max: 1, grid: { display: false }, ticks: { color: c.muted } };
    a.chart('', { data: { labels: labels, datasets: ds }, options: o });
  });
  // line-data: { x: [...], series: [{ label, y: [...] }], xlabel, ylabel }
  reg('line-data', function (a, p, c) {
    var pal = [c.c1, c.c2, c.c3, c.muted], xs = (p.x || []).map(Number);
    var ds = (p.series || []).map(function (s, i) { return a.lineDs(s.label || '', pts(xs, s.y.map(Number)), pal[i % 4], { pointRadius: 2 }); });
    a.chart('', { data: { datasets: ds }, options: opts(a, c, { legend: ds.length > 1, ymin: p.ymin != null ? p.ymin : null, xTitle: p.xlabel, yTitle: p.ylabel }) });
  });

  // ---------- 第12章 ----------
  // obs-exp-bar: { labels, observed, expected, ddof } 観測度数と期待度数、カイ二乗適合度
  reg('obs-exp-bar', function (a, p, c) {
    var O = (p.observed || []).map(Number), E = (p.expected || []).map(Number), L = p.labels || O.map(function (_, i) { return String(i + 1); });
    var contrib = O.map(function (o, i) { return Math.pow(o - E[i], 2) / E[i]; }), chi = contrib.reduce(function (s, v) { return s + v; }, 0);
    var df = O.length - 1 - num(p.ddof, 0), pv = df > 0 ? 1 - Dist.cdf('chi2', chi, { df: df }) : NaN;
    var ds = [barDs('観測度数 O', O, c, { barPercentage: 0.8, categoryPercentage: 0.8 }),
      barDs('期待度数 E', E, c, { backgroundColor: c.fill2, borderColor: c.c2, barPercentage: 0.8, categoryPercentage: 0.8 })];
    a.chart('', { data: { labels: L, datasets: ds }, options: opts(a, c, { xType: 'category', legend: true, yTitle: '度数' }) });
    a.info.textContent = '各セルの (O−E)²/E = ' + contrib.map(function (v) { return a.fmt(v, 3); }).join(', ') + '　χ² = ' + a.fmt(chi, 3) + '（自由度 ' + df + '，p値 = ' + a.fmt(pv) + '）';
  });
  // ks-ecdf: { data: [...], dist, その分布のパラメータ } 経験分布関数と理論分布、KS統計量
  reg('ks-ecdf', function (a, p, c) {
    var dn = p.dist || 'norm', q = distParams(dn, p), x = sorted((p.data || []).map(Number)), n = x.length;
    if (!n) { a.info.textContent = 'data がありません'; return; }
    var D = 0, at = x[0], yl = 0, yh = 0;
    x.forEach(function (v, i) {
      var F = Dist.cdf(dn, v, q), d1 = (i + 1) / n - F, d2 = F - i / n;
      if (d1 > D) { D = d1; at = v; yl = F; yh = (i + 1) / n; }
      if (d2 > D) { D = d2; at = v; yl = i / n; yh = F; }
    });
    var lo = Math.min(x[0], Dist.quantile(dn, 0.005, q)), hi = Math.max(x[n - 1], Dist.quantile(dn, 0.995, q)), xs = a.linspace(lo, hi, 241);
    var step = [{ x: lo, y: 0 }]; x.forEach(function (v, i) { step.push({ x: v, y: i / n }, { x: v, y: (i + 1) / n }); }); step.push({ x: hi, y: 1 });
    a.chart('', { data: { datasets: [a.lineDs('経験分布関数 Fn(x)', step, c.c1), a.lineDs('理論の F(x)', pts(xs, xs.map(function (v) { return Dist.cdf(dn, v, q); })), c.c2, { borderDash: [6, 4] }),
      a.lineDs('最大のずれ D', [{ x: at, y: yl }, { x: at, y: yh }], c.fg, { borderWidth: 3 })] },
    options: opts(a, c, { legend: true, xmin: lo, xmax: hi, ymax: 1, xTitle: 'x' }) });
    // 漸近分布（スティーブンスの補正付き）による p 値の近似
    var t = (Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n)) * D, pv = 0;
    for (var k = 1; k < 100; k++) pv += 2 * Math.pow(-1, k - 1) * Math.exp(-2 * k * k * t * t);
    a.info.textContent = 'D = ' + a.fmt(D) + '（x = ' + a.fmt(at) + ' で最大）　n = ' + n + '　p値 ≈ ' + a.fmt(Math.min(1, Math.max(0, pv))) + '（漸近近似）';
  });

  // ---------- 第20章 ----------
  // スチューデント化された範囲 q(k, df) の分布関数（数値積分）
  function phiFast(x) { var t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2); return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y); }
  function rangeCdf(w, k) {
    if (w <= 0) return 0;
    var n = 160, lo = -8, h = 16 / n, s = 0;
    for (var i = 0; i <= n; i++) { var z = lo + i * h, v = k * Dist.special.normPdf(z) * Math.pow(Math.max(0, phiFast(z) - phiFast(z - w)), k - 1); s += v * (i === 0 || i === n ? 1 : (i % 2 ? 4 : 2)); }
    return s * h / 3;
  }
  function srangeCdf(q, k, df) {
    if (q <= 0) return 0;
    var lc = (df / 2) * Math.log(df) - (df / 2 - 1) * Math.LN2 - Dist.special.lgamma(df / 2), smax = 1 + 10 / Math.sqrt(df), n = 120, h = smax / n, s = 0;
    for (var i = 1; i <= n; i++) { var x = i * h, g = Math.exp(lc + (df - 1) * Math.log(x) - df * x * x / 2); s += g * rangeCdf(q * x, k) * (i === n ? 1 : (i % 2 ? 4 : 2)); }
    return Math.min(1, s * h / 3);
  }
  Charts._srangeCdf = srangeCdf;
  // studentized-range: { k, df, alpha } スチューデント化された範囲の分布と上側 α 点
  reg('studentized-range', function (a, p, c) {
    var k = Math.max(2, Math.round(num(p.k, 3))), df = Math.max(1, Math.round(num(p.df, 9))), al = num(p.alpha, 0.05);
    var lo = 0, hi = 12;
    for (var it = 0; it < 40; it++) { var mid = (lo + hi) / 2; if (srangeCdf(mid, k, df) < 1 - al) lo = mid; else hi = mid; }
    var qa = (lo + hi) / 2, xmax = Math.max(6, qa * 1.6), xs = a.linspace(0, xmax, 81), F = xs.map(function (x) { return srangeCdf(x, k, df); });
    var f = xs.map(function (x, i) { var j = Math.min(Math.max(i, 1), xs.length - 2); return i === 0 ? 0 : Math.max(0, (F[j + 1] - F[j - 1]) / (xs[j + 1] - xs[j - 1])); });
    a.chart('', { data: { datasets: [a.lineDs('q(k, df) の密度', pts(xs, f), c.c1, { tension: 0.3 }),
      a.lineDs('上側 α', pts(xs, xs.map(function (x, i) { return x >= qa ? f[i] : null; })), c.c2, { fill: 'origin', backgroundColor: c.fill2, borderWidth: 0, tension: 0.3 }),
      a.lineDs('q_α', [{ x: qa, y: 0 }, { x: qa, y: Math.max.apply(null, f) }], c.c2, { borderDash: [5, 4], borderWidth: 1.5 })] },
    options: opts(a, c, { legend: true, xmin: 0, xmax: xmax, xTitle: 'q' }) });
    a.info.textContent = 'k = ' + k + '，自由度 ' + df + ' の上側 ' + a.fmt(al) + ' 点 q = ' + a.fmt(qa, 3) + '（数値積分による値）';
  });

  // ---------- 第22・25章 ----------
  // points-2d: { points: [{label, x, y}], arrows: 1, circle: 1, xlabel, ylabel } ラベル付きの点（負荷量プロット・得点の散布図）
  reg('points-2d', function (a, p, c) {
    var P = p.points || [], xs = P.map(function (q) { return Number(q.x); }), ys = P.map(function (q) { return Number(q.y); });
    var lim = Math.max(p.circle ? 1.1 : 0, Math.max.apply(null, xs.concat(ys).map(Math.abs).concat([0.1])) * 1.15);
    var cv = a.canvas(320), ctx = cv.ctx, W = cv.w, H = cv.h, pad = 34, S = Math.min(W - 2 * pad, H - 2 * pad) / (2 * lim), cx = W / 2, cy = H / 2;
    var X = function (v) { return cx + v * S; }, Y = function (v) { return cy - v * S; };
    ctx.strokeStyle = c.line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(-lim), cy); ctx.lineTo(X(lim), cy); ctx.moveTo(cx, Y(-lim)); ctx.lineTo(cx, Y(lim)); ctx.stroke();
    ctx.fillStyle = c.muted; ctx.font = '11px ' + c.font; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    [-lim, -lim / 2, lim / 2, lim].forEach(function (v) { ctx.fillText(a.fmt(v, 2), X(v), cy + 4); });
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    [-lim, -lim / 2, lim / 2, lim].forEach(function (v) { ctx.fillText(a.fmt(v, 2), cx - 4, Y(v)); });
    if (p.circle) { ctx.strokeStyle = c.faint; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.arc(cx, cy, S, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]); }
    ctx.font = '12px ' + c.font;
    var boxes = [], groups = [], pal = [c.c1, c.c2, c.c3, c.muted];
    // 点の種類（q.group か、ラベル先頭の [行] などの印）ごとに色を変える
    var grp = function (q) { var m = /^\[(.+?)\]/.exec(String(q.label || '')); var g = q.group != null ? String(q.group) : (m ? m[1] : ''); if (groups.indexOf(g) < 0) groups.push(g); return groups.indexOf(g); };
    P.forEach(function (q, i) {
      var x = X(xs[i]), y = Y(ys[i]), col = pal[grp(q) % 4];
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.6;
      if (p.arrows) drawArrow(ctx, cx, cy, x, y, 8);
      else { ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill(); }
    });
    // ラベルは重ならないよう、ぶつかったら縦にずらす
    P.forEach(function (q, i) {
      var lab = q.label != null ? String(q.label) : '', w = ctx.measureText(lab).width, right = xs[i] >= 0;
      var lx = X(xs[i]) + (right ? 5 : -5 - w), ly = Y(ys[i]) + (ys[i] >= 0 ? -17 : 3), dir = ys[i] >= 0 ? -1 : 1;
      for (var t = 0; t < 12 && boxes.some(function (b) { return lx < b[0] + b[2] && lx + w > b[0] && ly < b[1] + 14 && ly + 14 > b[1]; }); t++) ly += dir * 14;
      boxes.push([lx, ly, w]);
      ctx.fillStyle = c.fg; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(lab, lx, ly);
    });
    ctx.fillStyle = c.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(p.xlabel || '', W - 6, cy - 4);
    ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(p.ylabel || '', cx + 6, 4);
  });
  // graph-diagram: { nodes: [...], edges: [[i, j], ...], directed: 1 で矢印, weights: [...]（辺のラベル、省略可）, layout: circle|line }
  reg('graph-diagram', graphDiagram);
  // graph-network: graph-diagram と同じ（edgeLabels を辺のラベル、layout: triangle は circle として扱う）
  reg('graph-network', function (a, p, c) { graphDiagram(a, Object.assign({}, p, { weights: p.weights || p.edgeLabels, layout: p.layout === 'line' ? 'line' : 'circle' }), c); });
  function graphDiagram(a, p, c) {
    var N = p.nodes || ['X1', 'X2', 'X3'], E = p.edges || [], K = N.length, cv = a.canvas(240), ctx = cv.ctx, W = cv.w, H = cv.h, r = 22;
    var pos = N.map(function (_, i) {
      if (p.layout === 'line') return [40 + (W - 80) * (K === 1 ? 0.5 : i / (K - 1)), H / 2];
      var t = -Math.PI / 2 + 2 * Math.PI * i / K, R = Math.min(W, H) / 2 - 42;
      return [W / 2 + R * Math.cos(t), H / 2 + R * Math.sin(t)];
    });
    ctx.lineWidth = 1.8; ctx.font = '12px ' + c.font;
    var rad = N.map(function (lab) { return Math.max(r, ctx.measureText(String(lab)).width / 2 + 9); });
    E.forEach(function (e, k) {
      var i = e[0], j = e[1], dx = pos[j][0] - pos[i][0], dy = pos[j][1] - pos[i][1], d = Math.hypot(dx, dy), ux = dx / d, uy = dy / d;
      var bend = p.layout === 'line' && Math.abs(j - i) > 1 ? -40 : 0;
      ctx.strokeStyle = c.muted; ctx.fillStyle = c.muted;
      var x1 = pos[i][0] + ux * rad[i], y1 = pos[i][1] + uy * rad[i], x2 = pos[j][0] - ux * rad[j], y2 = pos[j][1] - uy * rad[j];
      if (bend) {
        ctx.beginPath(); ctx.moveTo(pos[i][0], pos[i][1] - r); ctx.quadraticCurveTo((pos[i][0] + pos[j][0]) / 2, pos[i][1] + 2 * bend, pos[j][0], pos[j][1] - r); ctx.stroke();
        if (p.directed) drawArrow(ctx, pos[j][0] - 6, pos[j][1] - r - 10, pos[j][0], pos[j][1] - r, 8);
      } else if (p.directed) drawArrow(ctx, x1, y1, x2, y2, 9);
      else { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
      if (p.weights && p.weights[k] != null) {
        ctx.fillStyle = c.fg; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(String(p.weights[k]), (pos[i][0] + pos[j][0]) / 2 - uy * 12, (pos[i][1] + pos[j][1]) / 2 + ux * 12 + bend);
      }
    });
    pos.forEach(function (q, i) {
      ctx.fillStyle = c.surface; ctx.strokeStyle = c.c1; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(q[0], q[1], rad[i], 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = c.fg; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(N[i]), q[0], q[1]);
    });
    a.info.textContent = p.directed ? '矢印は直接の影響（親 → 子）' : '辺がない2頂点は、他のすべての変数を与えたもとで条件付き独立';
  }

  // ---------- 第23章 ----------
  // 座標平面（軸つき）を描く道具
  function plane(a, c, xmin, xmax, ymin, ymax, height) {
    var cv = a.canvas(height || 320), ctx = cv.ctx, W = cv.w, H = cv.h, pad = 30;
    var sc = Math.min((W - 2 * pad) / (xmax - xmin), (H - 2 * pad) / (ymax - ymin)), ox = (W - sc * (xmax - xmin)) / 2, oy = (H - sc * (ymax - ymin)) / 2;
    var X = function (v) { return ox + (v - xmin) * sc; }, Y = function (v) { return H - oy - (v - ymin) * sc; };
    ctx.strokeStyle = c.line; ctx.lineWidth = 1; ctx.strokeRect(X(xmin), Y(ymax), (xmax - xmin) * sc, (ymax - ymin) * sc);
    ctx.fillStyle = c.muted; ctx.font = '11px ' + c.font; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var t = Math.ceil(xmin); t <= xmax; t++) if (t % Math.max(1, Math.round((xmax - xmin) / 8)) === 0) ctx.fillText(String(t), X(t), Y(ymin) + 3);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (t = Math.ceil(ymin); t <= ymax; t++) if (t % Math.max(1, Math.round((ymax - ymin) / 8)) === 0) ctx.fillText(String(t), X(xmin) - 3, Y(t));
    return { ctx: ctx, X: X, Y: Y, sc: sc, W: W, H: H, xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax };
  }
  function dot(P, x, y, col, r, shape) {
    var ctx = P.ctx; ctx.fillStyle = col; ctx.strokeStyle = col;
    if (shape === 'x') { ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(P.X(x) - r, P.Y(y) - r); ctx.lineTo(P.X(x) + r, P.Y(y) + r); ctx.moveTo(P.X(x) - r, P.Y(y) + r); ctx.lineTo(P.X(x) + r, P.Y(y) - r); ctx.stroke(); }
    else { ctx.beginPath(); ctx.arc(P.X(x), P.Y(y), r, 0, 2 * Math.PI); ctx.fill(); }
  }
  function shade(P, f, col1, col2) {
    var ctx = P.ctx, step = 6;
    for (var px = P.X(P.xmin); px < P.X(P.xmax); px += step) for (var py = P.Y(P.ymax); py < P.Y(P.ymin); py += step) {
      var x = P.xmin + (px + step / 2 - P.X(P.xmin)) / P.sc, y = P.ymax - (py + step / 2 - P.Y(P.ymax)) / P.sc;
      ctx.fillStyle = f(x, y) ? col2 : col1; ctx.fillRect(px, py, step, step);
    }
  }
  function twoGroups(a, n, d1, d2, rho, ratio) {
    var G = [[], []], s = Math.sqrt(ratio || 1);
    for (var i = 0; i < n; i++) {
      var g = i % 2, u = a.normal(), v = rho * u + Math.sqrt(1 - rho * rho) * a.normal(), k = g ? s : 1;
      G[g].push([(g ? d1 : 0) + k * u, (g ? d2 : 0) + k * v]);
    }
    return G;
  }
  // discriminant-boundary: { delta, rho, ratio, prior, n } 2つの正規分布のベイズ判別境界（ratio=1 で線形、それ以外は二次）
  reg('discriminant-boundary', function (a, p, c) {
    var dl = num(p.delta, 2), rho = num(p.rho, 0.5), ratio = num(p.ratio, 1), pr = num(p.prior, 0.5), n = Math.round(num(p.n, 60));
    var G = twoGroups(a, n, dl, 0, rho, ratio), det = 1 - rho * rho;
    var q = function (x, y, s2) { return (x * x - 2 * rho * x * y + y * y) / (det * s2); };
    var score = function (x, y) { // log f2 + log π2 − log f1 − log π1
      return -0.5 * q(x - dl, y, ratio) - Math.log(ratio) + 0.5 * q(x, y, 1) + Math.log(pr / (1 - pr));
    };
    var lim = Math.max(4, dl + 3), P = plane(a, c, -3.5, lim, -3.5, 3.5, 300);
    shade(P, function (x, y) { return score(x, y) > 0; }, c.fill, c.fill2);
    G[0].forEach(function (q2) { dot(P, q2[0], q2[1], c.c1, 3); });
    G[1].forEach(function (q2) { dot(P, q2[0], q2[1], c.c2, 4, 'x'); });
    var err = 0; G[0].forEach(function (q2) { if (score(q2[0], q2[1]) > 0) err++; }); G[1].forEach(function (q2) { if (score(q2[0], q2[1]) <= 0) err++; });
    a.info.textContent = (ratio === 1 ? '共分散が等しいので境界は直線（線形判別）' : '共分散が異なるので境界は二次曲線（二次判別）') + '　群2の事前確率 ' + a.fmt(pr) + '　この標本の誤判別 ' + err + '/' + n + '（● 群1，× 群2，背景色＝判別される群）';
  });
  // fisher-projection: { angle（度）, rho, delta, n } 射影の向きと、射影した値の2群の分布
  reg('fisher-projection', function (a, p, c) {
    var ang = num(p.angle, 0) * Math.PI / 180, rho = num(p.rho, 0.6), dl = num(p.delta, 2), n = Math.round(num(p.n, 60)), G = twoGroups(a, n, dl, 0, rho, 1);
    var w = [Math.cos(ang), Math.sin(ang)], J = function (v) { var num1 = Math.pow(v[0] * dl, 2), den = v[0] * v[0] + 2 * rho * v[0] * v[1] + v[1] * v[1]; return num1 / den; };
    var opt = [1 / (1 - rho * rho) * dl, -rho / (1 - rho * rho) * dl], on = Math.hypot(opt[0], opt[1]); opt = [opt[0] / on, opt[1] / on];
    var P = plane(a, c, -3.5, dl + 3.5, -3.5, 3.5, 260), ctx = P.ctx, mx = dl / 2;
    G[0].forEach(function (q) { dot(P, q[0], q[1], c.c1, 3); });
    G[1].forEach(function (q) { dot(P, q[0], q[1], c.c2, 4, 'x'); });
    ctx.strokeStyle = c.fg; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(P.X(mx - 4 * w[0]), P.Y(-4 * w[1])); ctx.lineTo(P.X(mx + 4 * w[0]), P.Y(4 * w[1])); ctx.stroke();
    ctx.setLineDash([5, 4]); ctx.strokeStyle = c.c3; ctx.beginPath(); ctx.moveTo(P.X(mx - 4 * opt[0]), P.Y(-4 * opt[1])); ctx.lineTo(P.X(mx + 4 * opt[0]), P.Y(4 * opt[1])); ctx.stroke(); ctx.setLineDash([]);
    var proj = G.map(function (g) { return g.map(function (q) { return q[0] * w[0] + q[1] * w[1]; }); }), all = proj[0].concat(proj[1]);
    var lo = Math.min.apply(null, all) - 0.5, hi = Math.max.apply(null, all) + 0.5;
    a.chart('short', { data: { datasets: [barDs('群1を射影', a.hist(proj[0], 24, lo, hi), c), barDs('群2を射影', a.hist(proj[1], 24, lo, hi), c, { backgroundColor: c.fill2, borderColor: c.c2 })] },
      options: opts(a, c, { legend: true, xmin: lo, xmax: hi, xTitle: '射影した値 w′x' }) });
    a.info.textContent = '群間／群内の比 J(w) = ' + a.fmt(J(w), 3) + '（最大値 ' + a.fmt(J(opt), 3) + ' は点線の向き S⁻¹d：角度 ' + a.fmt(Math.atan2(opt[1], opt[0]) * 180 / Math.PI, 1) + '°）';
  });
  // mahalanobis-contour: { rho, sx, sy } マハラノビス距離 1,2,3 の楕円とユークリッド距離の円
  reg('mahalanobis-contour', function (a, p, c) {
    var rho = num(p.rho, 0.7), sx = num(p.sx, 1), sy = num(p.sy, 1), lim = 3.6 * Math.max(sx, sy), P = plane(a, c, -lim, lim, -lim, lim, 320), ctx = P.ctx;
    var L = [[sx, 0], [rho * sy, sy * Math.sqrt(1 - rho * rho)]];
    [1, 2, 3].forEach(function (k) {
      ctx.strokeStyle = c.c1; ctx.lineWidth = 2; ctx.beginPath();
      for (var t = 0; t <= 100; t++) { var th = 2 * Math.PI * t / 100, u = k * Math.cos(th), v = k * Math.sin(th), x = L[0][0] * u, y = L[1][0] * u + L[1][1] * v; t ? ctx.lineTo(P.X(x), P.Y(y)) : ctx.moveTo(P.X(x), P.Y(y)); }
      ctx.stroke();
      ctx.strokeStyle = c.faint; ctx.lineWidth = 1; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.arc(P.X(0), P.Y(0), k * P.sc, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
    });
    var md = function (x, y) { var a1 = x / sx, b1 = y / sy; return Math.sqrt((a1 * a1 - 2 * rho * a1 * b1 + b1 * b1) / (1 - rho * rho)); };
    var pts2 = [[1.5, 1.5, 'A'], [1.5, -1.5, 'B']];
    ctx.font = '12px ' + c.font;
    pts2.forEach(function (q) { dot(P, q[0], q[1], c.c2, 5); ctx.fillStyle = c.fg; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(q[2], P.X(q[0]) + 6, P.Y(q[1]) - 4); });
    a.info.textContent = '実線＝マハラノビス距離 1,2,3，点線＝ユークリッド距離 1,2,3　A(1.5, 1.5)：ユークリッド ' + a.fmt(Math.hypot(1.5, 1.5), 3) + '，マハラノビス ' + a.fmt(md(1.5, 1.5), 3) + '　B(1.5, −1.5)：ユークリッド ' + a.fmt(Math.hypot(1.5, 1.5), 3) + '，マハラノビス ' + a.fmt(md(1.5, -1.5), 3);
  });
  // knn-boundary: { k, n } k近傍法の判別領域
  reg('knn-boundary', function (a, p, c) {
    var k = Math.max(1, Math.round(num(p.k, 5))), n = Math.round(num(p.n, 60)), G = twoGroups(a, n, 1.6, 1.2, 0, 1), D = [];
    G.forEach(function (g, gi) { g.forEach(function (q) { D.push([q[0], q[1], gi]); }); });
    var P = plane(a, c, -3, 4.5, -3, 4.2, 300);
    var cls = function (x, y) {
      var d = D.map(function (q) { return [Math.pow(q[0] - x, 2) + Math.pow(q[1] - y, 2), q[2]]; }).sort(function (u, v) { return u[0] - v[0]; });
      var s = 0; for (var i = 0; i < k; i++) s += d[i][1]; return s > k / 2;
    };
    shade(P, cls, c.fill, c.fill2);
    G[0].forEach(function (q) { dot(P, q[0], q[1], c.c1, 3); });
    G[1].forEach(function (q) { dot(P, q[0], q[1], c.c2, 4, 'x'); });
    var err = 0; D.forEach(function (q, i) {
      var d = D.map(function (r, j) { return [j === i ? Infinity : Math.pow(r[0] - q[0], 2) + Math.pow(r[1] - q[1], 2), r[2]]; }).sort(function (u, v) { return u[0] - v[0]; });
      var s = 0; for (var t = 0; t < k; t++) s += d[t][1]; if ((s > k / 2 ? 1 : 0) !== q[2]) err++;
    });
    a.info.textContent = 'k = ' + k + '：k が小さいと境界は入り組み（過学習）、大きいとなめらかになる　一つ抜き誤判別 ' + err + '/' + D.length;
  });
  // svm-margin: { gap, n } マージン最大の分離直線とサポートベクトル
  reg('svm-margin', function (a, p, c) {
    var gap = num(p.gap, 1), n = Math.round(num(p.n, 30)), D = [];
    for (var i = 0; i < n; i++) { var g = i % 2 ? 1 : -1, u = a.normal(), v = a.normal(); D.push([0.6 * u + g * (0.9 + gap / 2) * 0.7071 + 0.8, 0.6 * v + g * (0.9 + gap / 2) * 0.7071 + 0.8, g]); }
    // 向き θ を細かく試し、射影したときの2群のすきまが最大になる向きを探す（2次元なのでこれで最適解に十分近い）
    var best = { m: -Infinity };
    for (var t = 0; t < 3600; t++) {
      var th = 2 * Math.PI * t / 3600, w = [Math.cos(th), Math.sin(th)], lo = Infinity, hi = -Infinity;
      D.forEach(function (q) { var z = q[0] * w[0] + q[1] * w[1]; if (q[2] > 0) lo = Math.min(lo, z); else hi = Math.max(hi, z); });
      if ((lo - hi) / 2 > best.m) best = { m: (lo - hi) / 2, w: w, b: (lo + hi) / 2 };
    }
    var P = plane(a, c, -3.5, 5, -3.5, 5, 320), ctx = P.ctx, w = best.w;
    var line = function (off, col, dash, lw) {
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(dash);
      var c0 = best.b + off, x1 = -10, x2 = 10;
      ctx.beginPath();
      if (Math.abs(w[1]) > 1e-6) { ctx.moveTo(P.X(x1), P.Y((c0 - w[0] * x1) / w[1])); ctx.lineTo(P.X(x2), P.Y((c0 - w[0] * x2) / w[1])); }
      else { ctx.moveTo(P.X(c0 / w[0]), P.Y(-10)); ctx.lineTo(P.X(c0 / w[0]), P.Y(10)); }
      ctx.save(); ctx.beginPath(); ctx.rect(P.X(P.xmin), P.Y(P.ymax), (P.xmax - P.xmin) * P.sc, (P.ymax - P.ymin) * P.sc); ctx.clip();
      ctx.beginPath();
      if (Math.abs(w[1]) > 1e-6) { ctx.moveTo(P.X(x1), P.Y((c0 - w[0] * x1) / w[1])); ctx.lineTo(P.X(x2), P.Y((c0 - w[0] * x2) / w[1])); }
      else { ctx.moveTo(P.X(c0 / w[0]), P.Y(-10)); ctx.lineTo(P.X(c0 / w[0]), P.Y(10)); }
      ctx.stroke(); ctx.restore(); ctx.setLineDash([]);
    };
    var sep = best.m > 0;
    line(0, c.fg, [], 2);
    if (sep) { line(best.m, c.muted, [5, 4], 1.2); line(-best.m, c.muted, [5, 4], 1.2); }
    D.forEach(function (q) {
      var z = q[0] * w[0] + q[1] * w[1] - best.b, sv = sep && Math.abs(Math.abs(z) - best.m) < 1e-6 + best.m * 0.01;
      dot(P, q[0], q[1], q[2] > 0 ? c.c2 : c.c1, q[2] > 0 ? 4 : 3, q[2] > 0 ? 'x' : null);
      if (sv) { ctx.strokeStyle = c.fg; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(P.X(q[0]), P.Y(q[1]), 8, 0, 2 * Math.PI); ctx.stroke(); }
    });
    a.info.textContent = sep ? 'マージン（境界から最も近い点までの距離）= ' + a.fmt(best.m, 3) + '　丸で囲んだ点がサポートベクトル（破線上の点）' : '2群が重なっていて直線では分離できない（ハードマージンでは解がない。ソフトマージンでは一部のはみ出しを許して境界を決める）。実線は重なりが最小の向き';
  });

  // minkowski-ball: { p } ミンコフスキー距離で原点から1の点の集まり（単位円）
  reg('minkowski-ball', function (a, p, c) {
    var q = Math.max(1, num(p.p, 2)), P = plane(a, c, -1.4, 1.4, -1.4, 1.4, 300), ctx = P.ctx;
    var curve = function (pp, col, dash, lw) {
      ctx.strokeStyle = col; ctx.setLineDash(dash); ctx.lineWidth = lw; ctx.beginPath();
      for (var t = 0; t <= 400; t++) {
        var th = 2 * Math.PI * t / 400, cs = Math.cos(th), sn = Math.sin(th), r = isFinite(pp) ? Math.pow(Math.pow(Math.abs(cs), pp) + Math.pow(Math.abs(sn), pp), -1 / pp) : 1 / Math.max(Math.abs(cs), Math.abs(sn));
        t ? ctx.lineTo(P.X(r * cs), P.Y(r * sn)) : ctx.moveTo(P.X(r * cs), P.Y(r * sn));
      }
      ctx.stroke(); ctx.setLineDash([]);
    };
    curve(1, c.faint, [3, 3], 1); curve(2, c.faint, [3, 3], 1); curve(Infinity, c.faint, [3, 3], 1);
    curve(q, c.c1, [], 2.5);
    ctx.strokeStyle = c.line; ctx.beginPath(); ctx.moveTo(P.X(-1.4), P.Y(0)); ctx.lineTo(P.X(1.4), P.Y(0)); ctx.moveTo(P.X(0), P.Y(-1.4)); ctx.lineTo(P.X(0), P.Y(1.4)); ctx.stroke();
    a.info.textContent = 'p = ' + a.fmt(q, 2) + '（実線）　点線は p = 1（マンハッタン：ひし形）、p = 2（ユークリッド：円）、p = ∞（チェビシェフ：正方形）　点 (1,1) までの距離 = ' + a.fmt(Math.pow(2, 1 / q), 4);
  });

  // rotation-angle: { loadings: [[a, b], ...], labels, angle（度） } 因子負荷量を固定して座標軸を回し、バリマックス基準を表示
  reg('rotation-angle', function (a, p, c) {
    var L = p.loadings || [[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]], lab = p.labels || L.map(function (_, i) { return 'x' + (i + 1); });
    var th = num(p.angle, 0) * Math.PI / 180, cs = Math.cos(th), sn = Math.sin(th);
    var R = L.map(function (r) { return [r[0] * cs + r[1] * sn, -r[0] * sn + r[1] * cs]; }), m = L.length;
    var V = [0, 1].reduce(function (s, k) { var s2 = 0, s4 = 0; R.forEach(function (r) { s2 += r[k] * r[k]; s4 += Math.pow(r[k], 4); }); return s + s4 / m - Math.pow(s2 / m, 2); }, 0);
    var P = plane(a, c, -1.1, 1.1, -1.1, 1.1, 300), ctx = P.ctx;
    ctx.strokeStyle = c.line; ctx.beginPath(); ctx.moveTo(P.X(-1.1), P.Y(0)); ctx.lineTo(P.X(1.1), P.Y(0)); ctx.moveTo(P.X(0), P.Y(-1.1)); ctx.lineTo(P.X(0), P.Y(1.1)); ctx.stroke();
    ctx.lineWidth = 2;
    [[cs, sn, '第1因子（回転後）', c.c2], [-sn, cs, '第2因子（回転後）', c.c3]].forEach(function (ax) {
      ctx.strokeStyle = ax[3]; ctx.beginPath(); ctx.moveTo(P.X(-ax[0]), P.Y(-ax[1])); ctx.lineTo(P.X(ax[0]), P.Y(ax[1])); ctx.stroke();
      ctx.fillStyle = ax[3]; ctx.font = '11px ' + c.font; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(ax[2], P.X(ax[0] * 0.95), P.Y(ax[1] * 0.95) - 4);
    });
    ctx.font = '12px ' + c.font;
    L.forEach(function (r, i) { dot(P, r[0], r[1], c.c1, 4); ctx.fillStyle = c.fg; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(String(lab[i]), P.X(r[0]) + 6, P.Y(r[1]) - 2); });
    a.info.textContent = '回転角 ' + a.fmt(num(p.angle, 0)) + '°　回転後の負荷量：' + R.map(function (r, i) { return lab[i] + ' (' + a.fmt(r[0], 3) + ', ' + a.fmt(r[1], 3) + ')'; }).join('，') + '　バリマックス基準 V = ' + a.fmt(V, 4) + '（共通性は回転で変わらない）';
  });

  // ---------- 第2章 ----------
  // mgf-curve: { family: poisson|binom|norm|expon, lambda, n, p, mu, sigma, rate } M(t) と t=0 の接線
  reg('mgf-curve', function (a, p, c) {
    var fam = p.family || 'poisson', M, m1, lo = -1, hi = 1;
    if (fam === 'binom') { var n = num(p.n, 10), q = num(p.p, 0.3); M = function (t) { return Math.pow(1 - q + q * Math.exp(t), n); }; m1 = n * q; }
    else if (fam === 'norm') { var mu = num(p.mu, 0), s = num(p.sigma, 1); M = function (t) { return Math.exp(mu * t + s * s * t * t / 2); }; m1 = mu; }
    else if (fam === 'expon') { var r = num(p.rate, 1); M = function (t) { return t < r ? r / (r - t) : null; }; m1 = 1 / r; hi = Math.min(1, 0.9 * r); lo = -1; }
    else { var lam = num(p.lambda, 3); M = function (t) { return Math.exp(lam * (Math.exp(t) - 1)); }; m1 = lam; }
    var xs = a.linspace(lo, hi, 201);
    var ds = [a.lineDs('M(t)', pts(xs, xs.map(M)), c.c1), a.lineDs('t=0 での接線（傾き E[X]）', pts(xs, xs.map(function (t) { return 1 + m1 * t; })), c.c2, { borderDash: [6, 4] })];
    var ys = xs.map(M).filter(function (v) { return v != null; });
    a.chart('', { data: { datasets: ds }, options: opts(a, c, { legend: true, xmin: lo, xmax: hi, ymin: Math.min(0, Math.min.apply(null, ys)), ymax: Math.min(Math.max.apply(null, ys), 1 + Math.abs(m1) * 1.2 + 6), xTitle: 't' }) });
    a.info.textContent = "M(0) = 1，M'(0) = E[X] = " + a.fmt(m1);
  });

  // ---------- 第4章 ----------
  // logistic-curve: { beta0, beta1, xmin, xmax, probit: 1 で同じ傾きのプロビット曲線を重ねる }
  reg('logistic-curve', function (a, p, c) {
    var b0 = num(p.beta0, num(p.b0, 0)), b1 = num(p.beta1, num(p.b1, 1)), lo = num(p.xmin, -8), hi = num(p.xmax, 8), xs = a.linspace(lo, hi, 241);
    var ds = [a.lineDs('ロジスティック 1/(1+e^{-(β0+β1x)})', pts(xs, xs.map(function (x) { return 1 / (1 + Math.exp(-(b0 + b1 * x))); })), c.c1)];
    if (p.probit) ds.push(a.lineDs('プロビット Φ((β0+β1x)/1.702)', pts(xs, xs.map(function (x) { return Dist.special.normCdf((b0 + b1 * x) / 1.702); })), c.c2, { borderDash: [6, 4] }));
    a.chart('', { data: { datasets: ds }, options: opts(a, c, { legend: true, xmin: lo, xmax: hi, ymax: 1, xTitle: 'x', yTitle: 'p' }) });
    a.info.textContent = b1 !== 0 ? 'p = 0.5 となる x = −β0/β1 = ' + a.fmt(-b0 / b1) + '　その点での傾き β1/4 = ' + a.fmt(b1 / 4) : 'β1 = 0 なので p は一定';
  });
  // box-cox: { lambda } 変換 (x^λ−1)/λ の曲線（λ=−1,0,0.5,1,2 を薄く重ねる）
  reg('box-cox', function (a, p, c) {
    var lam = num(p.lambda, 0.5), xs = a.linspace(0.05, num(p.xmax, 4), 200);
    var f = function (l) { return function (x) { return Math.abs(l) < 1e-9 ? Math.log(x) : (Math.pow(x, l) - 1) / l; }; };
    var ds = [a.lineDs('λ = ' + a.fmt(lam, 2), pts(xs, xs.map(f(lam))), c.c1, { borderWidth: 3 })];
    [-1, 0, 1, 2].forEach(function (l) { ds.push(a.lineDs('λ = ' + l, pts(xs, xs.map(f(l))), c.faint, { borderWidth: 1, borderDash: [4, 4] })); });
    a.chart('', { data: { datasets: ds }, options: opts(a, c, { legend: true, xmin: 0, xmax: xs[xs.length - 1], ymin: -3, ymax: 4, xTitle: 'x', yTitle: 'y(λ)' }) });
    a.info.textContent = 'λ = 0 は log x、λ = 1 は x − 1（形は変わらない）。すべての曲線が x = 1 で 0 を通り、傾き 1 で交わる';
  });

  // ---------- 第7章 ----------
  // clt-sim: { base: expon|unif|bernoulli|poisson|chi2, n, reps, p, lambda } 標準化した標本平均のヒストグラム
  reg('clt-sim', function (a, p, c) {
    var base = p.base || 'expon', n = Math.max(1, Math.round(num(p.n, 5))), reps = Math.round(num(p.reps, 3000)), rand = a.rand, z = a.normal;
    var bp = num(p.p, 0.2), lam = num(p.lambda, 1), mu, sg, draw;
    if (base === 'unif') { mu = 0.5; sg = Math.sqrt(1 / 12); draw = rand; }
    else if (base === 'bernoulli') { mu = bp; sg = Math.sqrt(bp * (1 - bp)); draw = function () { return rand() < bp ? 1 : 0; }; }
    else if (base === 'poisson') { mu = lam; sg = Math.sqrt(lam); draw = function () { return poissonDraw(rand, lam); }; }
    else if (base === 'chi2') { mu = 1; sg = Math.SQRT2; draw = function () { var v = z(); return v * v; }; }
    else { mu = 1; sg = 1; draw = function () { return expDraw(rand, 1); }; }
    var vals = [];
    for (var r = 0; r < reps; r++) { var s = 0; for (var i = 0; i < n; i++) s += draw(); vals.push((s / n - mu) / (sg / Math.sqrt(n))); }
    var h = a.hist(vals, 48, -4, 4), xs = a.linspace(-4, 4, 161);
    a.chart('', { data: { datasets: [barDs('標準化した標本平均', h, c), a.lineDs('N(0,1)', pts(xs, xs.map(Dist.special.normPdf)), c.c2)] },
      options: opts(a, c, { legend: true, xmin: -4, xmax: 4, xTitle: '(X̄ − μ)/(σ/√n)' }) });
    var below = vals.filter(function (v) { return v <= 1.96; }).length / reps;
    a.info.textContent = 'n = ' + n + '，繰り返し ' + reps + ' 回　P(Z ≤ 1.96) の割合 = ' + a.fmt(below, 3) + '（正規なら 0.975）';
  });
  // lln-path: { p, n, paths } ベルヌーイ試行の標本平均の推移
  reg('lln-path', function (a, p, c) {
    var pr = num(p.p, 0.5), n = Math.round(num(p.n, 1000)), K = Math.round(num(p.paths, 5)), pal = [c.c1, c.c2, c.c3, c.muted, c.faint], ds = [];
    for (var k = 0; k < K; k++) {
      var s = 0, d = [];
      for (var i = 1; i <= n; i++) { s += a.rand() < pr ? 1 : 0; if (i < 50 || i % Math.max(1, Math.floor(n / 400)) === 0) d.push({ x: i, y: s / i }); }
      ds.push(a.lineDs('系列' + (k + 1), d, pal[k % 5], { borderWidth: 1.5 }));
    }
    ds.push(a.lineDs('p', [{ x: 1, y: pr }, { x: n, y: pr }], c.fg, { borderDash: [5, 4], borderWidth: 1 }));
    a.chart('', { data: { datasets: ds }, options: opts(a, c, { xmin: 1, xmax: n, ymax: 1, xTitle: '試行回数 n', yTitle: '標本平均' }) });
    a.info.textContent = 'n が大きくなると、どの系列も p = ' + a.fmt(pr) + ' に近づく（チェビシェフの不等式より、ずれの幅はおよそ 1/√n の速さで縮む）';
  });

  // ---------- 第9〜10章 ----------
  // ci-coverage: { mu, sigma, n, level, count, known: 1 で z区間（σ既知）、0 で t区間 }
  reg('ci-coverage', function (a, p, c) {
    var mu = num(p.mu, 0), sg = num(p.sigma, 1), n = Math.max(2, Math.round(num(p.n, 10))), lv = num(p.level, 0.95), K = Math.round(num(p.count, 100)), known = !!p.known;
    var q = known ? Dist.quantile('norm', 1 - (1 - lv) / 2, {}) : Dist.quantile('t', 1 - (1 - lv) / 2, { df: n - 1 });
    var iv = [], hit = 0;
    for (var k = 0; k < K; k++) {
      var xs = []; for (var i = 0; i < n; i++) xs.push(mu + sg * a.normal());
      var m = mean(xs), se = (known ? sg : sd(xs)) / Math.sqrt(n), lo = m - q * se, hi = m + q * se, ok = lo <= mu && mu <= hi;
      if (ok) hit++;
      iv.push([lo, hi, ok]);
    }
    var cv = a.canvas(Math.max(240, K * 3 + 40)), ctx = cv.ctx, W = cv.w, H = cv.h, pad = 30;
    var span = 4 * sg / Math.sqrt(n) * (known ? 1 : 1.3), X = function (v) { return pad + (W - 2 * pad) * (v - (mu - span)) / (2 * span); };
    var rowH = (H - 40) / K;
    iv.forEach(function (r, k) {
      ctx.strokeStyle = r[2] ? c.c1 : c.c2; ctx.lineWidth = r[2] ? 1.4 : 2.4;
      var y = 10 + k * rowH + rowH / 2;
      ctx.beginPath(); ctx.moveTo(Math.max(pad, X(r[0])), y); ctx.lineTo(Math.min(W - pad, X(r[1])), y); ctx.stroke();
    });
    ctx.strokeStyle = c.fg; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(mu), 4); ctx.lineTo(X(mu), H - 26); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = c.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('真の μ = ' + a.fmt(mu), X(mu), H - 22);
    a.info.textContent = (known ? 'z区間' : 't区間') + '（信頼係数 ' + a.fmt(lv) + '，n = ' + n + '）を ' + K + ' 本：μ を含んだのは ' + hit + ' 本（太い線が外れ）';
  });
  // power-curve: { mu0, mu1, sigma, n, alpha, alt: greater|two-sided } z検定の2種類の過誤と検出力関数
  reg('power-curve', function (a, p, c) {
    var mu0 = num(p.mu0, 0), mu1 = num(p.mu1, 0.5), sg = num(p.sigma, 1), n = Math.max(1, Math.round(num(p.n, 20))), al = num(p.alpha, 0.05), two = p.alt === 'two-sided';
    var se = sg / Math.sqrt(n), z = Dist.quantile('norm', 1 - (two ? al / 2 : al), {}), cu = mu0 + z * se, cl = mu0 - z * se;
    var power = function (m) { var r = 1 - Dist.cdf('norm', cu, { mu: m, sigma: se }); if (two) r += Dist.cdf('norm', cl, { mu: m, sigma: se }); return r; };
    var lo = Math.min(mu0, mu1) - 4 * se, hi = Math.max(mu0, mu1) + 4 * se, xs = a.linspace(lo, hi, 301);
    var f0 = function (x) { return Dist.pdf('norm', x, { mu: mu0, sigma: se }); }, f1 = function (x) { return Dist.pdf('norm', x, { mu: mu1, sigma: se }); };
    var rej = function (x) { return x >= cu || (two && x <= cl); };
    var ds = [
      a.lineDs('H0 のもとでの X̄', pts(xs, xs.map(f0)), c.c1),
      a.lineDs('H1 のもとでの X̄', pts(xs, xs.map(f1)), c.c2),
      a.lineDs('α（第1種の過誤）', pts(xs, xs.map(function (x) { return rej(x) ? f0(x) : null; })), c.c1, { fill: 'origin', backgroundColor: c.fill, borderWidth: 0 }),
      a.lineDs('β（第2種の過誤）', pts(xs, xs.map(function (x) { return rej(x) ? null : f1(x); })), c.c2, { fill: 'origin', backgroundColor: c.fill2, borderWidth: 0 })
    ];
    a.chart('short', { data: { datasets: ds }, options: opts(a, c, { legend: true, xmin: lo, xmax: hi, xTitle: 'x̄' }) });
    var ms = a.linspace(mu0 - 5 * se, mu0 + 5 * se, 201);
    a.chart('short', { data: { datasets: [a.lineDs('検出力 1−β(μ)', pts(ms, ms.map(power)), c.c1), a.lineDs('α', [{ x: ms[0], y: al }, { x: ms[200], y: al }], c.faint, { borderDash: [4, 4], borderWidth: 1 })] },
      options: opts(a, c, { legend: true, xmin: ms[0], xmax: ms[200], ymax: 1, xTitle: '真の μ' }) });
    a.info.textContent = '棄却限界 ' + (two ? a.fmt(cl) + ' と ' : '') + a.fmt(cu) + '　μ = ' + a.fmt(mu1) + ' での検出力 = ' + a.fmt(power(mu1)) + '，β = ' + a.fmt(1 - power(mu1));
  });

  // ---------- 第14〜15章 ----------
  // markov-chain: { P: [[...]], labels: [...], start: 0, steps: 20 } 状態遷移図と分布の推移
  reg('markov-chain', function (a, p, c) {
    var P = p.P || [[0.7, 0.3], [0.4, 0.6]], K = P.length, L = p.labels || P.map(function (_, i) { return String(i + 1); });
    var steps = Math.round(num(p.steps, 20)), start = Math.round(num(p.start, 0));
    var cv = a.canvas(260), ctx = cv.ctx, W = cv.w, H = cv.h, cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 66, r = 22;
    var pos = P.map(function (_, i) { var t = -Math.PI / 2 + 2 * Math.PI * i / K; return K === 2 ? [cx + (i ? 1 : -1) * Math.min(R * 1.4, W / 3), cy] : [cx + R * Math.cos(t), cy + R * Math.sin(t)]; });
    ctx.lineWidth = 1.4; ctx.font = '12px ' + c.font;
    for (var i = 0; i < K; i++) for (var j = 0; j < K; j++) {
      var pr = P[i][j]; if (!pr) continue;
      ctx.strokeStyle = c.muted; ctx.fillStyle = c.muted;
      if (i === j) {
        var ang = Math.atan2(pos[i][1] - cy, pos[i][0] - cx) || (i ? 0 : Math.PI), lx = pos[i][0] + Math.cos(ang) * (r + 12), ly = pos[i][1] + Math.sin(ang) * (r + 12);
        ctx.beginPath(); ctx.arc(lx, ly, 13, 0, 2 * Math.PI); ctx.stroke();
        ctx.fillStyle = c.fg; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(a.fmt(pr, 3), pos[i][0] + Math.cos(ang) * (r + 32), pos[i][1] + Math.sin(ang) * (r + 32));
        continue;
      }
      var dx = pos[j][0] - pos[i][0], dy = pos[j][1] - pos[i][1], d = Math.sqrt(dx * dx + dy * dy), ux = dx / d, uy = dy / d, ox = -uy * 7, oy = ux * 7;
      drawArrow(ctx, pos[i][0] + ux * r + ox, pos[i][1] + uy * r + oy, pos[j][0] - ux * r + ox, pos[j][1] - uy * r + oy, 8);
      ctx.fillStyle = c.fg; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(a.fmt(pr, 3), (pos[i][0] + pos[j][0]) / 2 + ox * 2.6, (pos[i][1] + pos[j][1]) / 2 + oy * 2.6);
    }
    pos.forEach(function (q, i) {
      ctx.fillStyle = c.surface; ctx.strokeStyle = c.c1; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = c.fg; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(L[i], q[0], q[1]);
    });
    // 分布の推移
    var pi = P.map(function (_, i) { return i === start ? 1 : 0; }), hist = [pi];
    for (var t = 0; t < steps; t++) { var nx = new Array(K).fill(0); for (i = 0; i < K; i++) for (j = 0; j < K; j++) nx[j] += pi[i] * P[i][j]; pi = nx; hist.push(pi); }
    var st = P.map(function () { return 1 / K; }); for (t = 0; t < 2000; t++) { var s2 = new Array(K).fill(0); for (i = 0; i < K; i++) for (j = 0; j < K; j++) s2[j] += st[i] * P[i][j]; st = s2.map(function (v, k) { return 0.5 * v + 0.5 * st[k]; }); }
    var pal = [c.c1, c.c2, c.c3, c.muted, c.faint];
    var ds = L.map(function (lab, k) { return a.lineDs('P(状態 ' + lab + ')', hist.map(function (h, t) { return { x: t, y: h[k] }; }), pal[k % 5], { pointRadius: 2 }); });
    a.chart('short', { data: { datasets: ds }, options: opts(a, c, { legend: true, xmin: 0, xmax: steps, ymax: 1, xTitle: 'ステップ n（状態 ' + L[start] + ' から出発）' }) });
    a.info.textContent = '定常分布 π ≈ (' + st.map(function (v) { return a.fmt(v, 4); }).join(', ') + ')';
  });
  // poisson-process: { lambda, T } 到着時刻と計数過程 N(t)
  reg('poisson-process', function (a, p, c) {
    var lam = num(p.lambda, 1), T = num(p.T, 20), t = 0, arr = [];
    while (true) { t += expDraw(a.rand, lam); if (t > T) break; arr.push(t); }
    var step = [{ x: 0, y: 0 }]; arr.forEach(function (s, i) { step.push({ x: s, y: i }); step.push({ x: s, y: i + 1 }); }); step.push({ x: T, y: arr.length });
    var ds = [a.lineDs('N(t)', step, c.c1), { type: 'scatter', label: '到着', data: arr.map(function (s) { return { x: s, y: 0 }; }), backgroundColor: c.c2, pointRadius: 4, pointStyle: 'triangle' }];
    a.chart('', { data: { datasets: ds }, options: opts(a, c, { legend: true, xmin: 0, xmax: T, xTitle: '時刻 t' }) });
    var gaps = arr.map(function (s, i) { return s - (i ? arr[i - 1] : 0); });
    a.info.textContent = '到着 ' + arr.length + ' 回（期待値 λT = ' + a.fmt(lam * T) + '）　到着間隔の平均 ' + a.fmt(gaps.length ? mean(gaps) : NaN) + '（理論値 1/λ = ' + a.fmt(1 / lam) + '）';
  });
  // random-walk: { p, steps, paths, lower, upper, start } 吸収壁（lower/upper）を指定すると破産問題
  reg('random-walk', function (a, p, c) {
    var pr = num(p.p, 0.5), n = Math.round(num(p.steps, 200)), K = Math.round(num(p.paths, 5)), lo = p.lower, hi = p.upper, s0 = num(p.start, 0);
    var pal = [c.c1, c.c2, c.c3, c.muted, c.faint], ds = [], hitU = 0, hitL = 0;
    for (var k = 0; k < K; k++) {
      var x = s0, d = [{ x: 0, y: x }];
      for (var i = 1; i <= n; i++) {
        x += a.rand() < pr ? 1 : -1; d.push({ x: i, y: x });
        if (hi != null && x >= hi) { hitU++; break; }
        if (lo != null && x <= lo) { hitL++; break; }
      }
      ds.push(a.lineDs('', d, pal[k % 5], { borderWidth: 1.4 }));
    }
    [lo, hi].forEach(function (b) { if (b != null) ds.push(a.lineDs('', [{ x: 0, y: b }, { x: n, y: b }], c.fg, { borderDash: [5, 4], borderWidth: 1 })); });
    a.chart('', { data: { datasets: ds }, options: opts(a, c, { xmin: 0, xmax: n, ymin: null, xTitle: 'ステップ' }) });
    var info = '上に進む確率 p = ' + a.fmt(pr) + '　期待値の傾き 2p − 1 = ' + a.fmt(2 * pr - 1);
    if (lo != null || hi != null) info += '　上の壁に到達 ' + hitU + ' 本，下の壁に到達 ' + hitL + ' 本（' + K + ' 本中）';
    a.info.textContent = info;
  });

  // ---------- 第16〜17章 ----------
  // regression-residual: { n, beta0, beta1, sigma, outlier: 最後の点（x が大きい点）の y のずれ }
  reg('regression-residual', function (a, p, c) {
    var n = Math.round(num(p.n, 20)), b0 = num(p.beta0, 1), b1 = num(p.beta1, 0.5), sg = num(p.sigma, 1), out = num(p.outlier, 0);
    var xs = [], ys = [];
    for (var i = 0; i < n - 1; i++) { var x = a.rand() * 10; xs.push(x); ys.push(b0 + b1 * x + sg * a.normal()); }
    xs.push(16); ys.push(b0 + b1 * 16 + out);
    var X = xs.map(function (x) { return [1, x]; }), bh = lsq(X, ys), bh2 = lsq(X.slice(0, n - 1), ys.slice(0, n - 1));
    var mx = mean(xs), sxx = 0; xs.forEach(function (x) { sxx += (x - mx) * (x - mx); });
    var res = ys.map(function (y, i) { return y - bh[0] - bh[1] * xs[i]; }), s2 = res.reduce(function (s, r) { return s + r * r; }, 0) / (n - 2);
    var h = xs.map(function (x) { return 1 / n + (x - mx) * (x - mx) / sxx; });
    var cook = res.map(function (r, i) { return r * r / (2 * s2) * h[i] / Math.pow(1 - h[i], 2); });
    var segs = [];
    xs.forEach(function (x, i) { segs.push({ x: x, y: ys[i] }, { x: x, y: bh[0] + bh[1] * x }, { x: NaN, y: NaN }); });
    var ds = [
      { type: 'scatter', label: 'データ', data: pts(xs.slice(0, n - 1), ys.slice(0, n - 1)), backgroundColor: c.c1, pointRadius: 3 },
      { type: 'scatter', label: 'てこ比の大きい点', data: [{ x: 16, y: ys[n - 1] }], backgroundColor: c.c2, pointRadius: 6, pointStyle: 'rectRot' },
      a.lineDs('残差', segs, c.faint, { borderWidth: 1, spanGaps: false }),
      a.lineDs('回帰直線（全データ）', [{ x: 0, y: bh[0] }, { x: 17, y: bh[0] + 17 * bh[1] }], c.c2),
      a.lineDs('回帰直線（その点を除く）', [{ x: 0, y: bh2[0] }, { x: 17, y: bh2[0] + 17 * bh2[1] }], c.c3, { borderDash: [6, 4] })
    ];
    a.chart('tall', { data: { datasets: ds }, options: opts(a, c, { legend: true, ymin: null, xmin: 0, xmax: 17, xTitle: 'x', yTitle: 'y' }) });
    a.info.textContent = '右端の点：てこ比 h = ' + a.fmt(h[n - 1], 3) + '（平均 2/n = ' + a.fmt(2 / n, 3) + '）　クックの距離 D = ' + a.fmt(cook[n - 1], 3) + '　傾き ' + a.fmt(bh[1], 3) + '（除くと ' + a.fmt(bh2[1], 3) + '）';
  });
  // residual-pattern: { pattern: none|funnel|curve|autocorr, strength, n } 残差 対 予測値のプロット
  reg('residual-pattern', function (a, p, c) {
    var pat = p.pattern || 'none', k = num(p.strength, 1), n = Math.round(num(p.n, 80)), xs = [], ys = [], e = 0;
    for (var i = 0; i < n; i++) {
      var x = 10 * (i + 0.5) / n, z = a.normal(), err;
      if (pat === 'funnel') err = z * (0.3 + k * 0.25 * x);
      else if (pat === 'autocorr') { e = Math.min(0.95, 0.3 * k) * e + z; err = e; }
      else err = z;
      var y = 2 + 1.5 * x + err + (pat === 'curve' ? k * 0.15 * Math.pow(x - 5, 2) : 0);
      xs.push(x); ys.push(y);
    }
    var b = lsq(xs.map(function (x) { return [1, x]; }), ys), fit = xs.map(function (x) { return b[0] + b[1] * x; }), res = ys.map(function (y, i) { return y - fit[i]; });
    var fmin = Math.min.apply(null, fit), fmax = Math.max.apply(null, fit);
    a.chart('', { data: { datasets: [{ type: 'scatter', label: '残差', data: pts(fit, res), backgroundColor: c.c1, pointRadius: 3 }, a.lineDs('0', [{ x: fmin, y: 0 }, { x: fmax, y: 0 }], c.c2, { borderDash: [5, 4] })] },
      options: opts(a, c, { ymin: null, xmin: fmin, xmax: fmax, xTitle: '予測値 ŷ', yTitle: '残差 e' }) });
    a.info.textContent = { none: '問題なし：0のまわりに一様な帯', funnel: '不均一分散：予測値が大きいほど残差が広がる（漏斗形）', curve: '非線形性：残差がU字を描く（2次の項などが必要）', autocorr: '系列相関：隣り合う残差が同じ符号になりやすい（並びの順に見る）' }[pat] || '';
  });
  // dw-sim: { phi, n } AR(1) 誤差をもつ回帰の残差とダービン・ワトソン比
  reg('dw-sim', function (a, p, c) {
    var phi = num(p.phi, 0.5), n = Math.round(num(p.n, 30)), e = 0, ts = [], ys = [];
    for (var t = 1; t <= n; t++) { e = phi * e + a.normal(); ts.push(t); ys.push(1 + 0.5 * t + e); }
    var b = lsq(ts.map(function (t) { return [1, t]; }), ys), r = ys.map(function (y, i) { return y - b[0] - b[1] * ts[i]; });
    var num1 = 0, den = 0, c1 = 0;
    for (var i = 0; i < n; i++) { den += r[i] * r[i]; if (i) { num1 += Math.pow(r[i] - r[i - 1], 2); c1 += r[i] * r[i - 1]; } }
    var d = num1 / den, r1 = c1 / den;
    a.chart('', { data: { datasets: [a.lineDs('残差（時間順）', pts(ts, r), c.c1, { pointRadius: 2.5 }), a.lineDs('0', [{ x: 1, y: 0 }, { x: n, y: 0 }], c.c2, { borderDash: [5, 4] })] },
      options: opts(a, c, { ymin: null, xmin: 1, xmax: n, xTitle: '時点 t', yTitle: '残差' }) });
    a.info.textContent = 'ダービン・ワトソン比 d = ' + a.fmt(d, 3) + '　残差のラグ1自己相関 r₁ = ' + a.fmt(r1, 3) + '　2(1−r₁) = ' + a.fmt(2 * (1 - r1), 3) + '（φ = ' + a.fmt(phi) + '）';
  });
  // iv-scatter: { rho, strength, n, beta } 内生性のある回帰で、最小二乗法と操作変数法の直線を比べる
  reg('iv-scatter', function (a, p, c) {
    var rho = num(p.rho, 0.6), st = num(p.strength, 1), n = Math.round(num(p.n, 200)), b = num(p.beta, 1), xs = [], ys = [], zs = [];
    for (var i = 0; i < n; i++) {
      var z = a.normal(), u = a.normal(), v = rho * u + Math.sqrt(Math.max(0, 1 - rho * rho)) * a.normal(), x = st * z + v;
      zs.push(z); xs.push(x); ys.push(b * x + u);
    }
    var mx = mean(xs), my = mean(ys), mz = mean(zs), sxx = 0, sxy = 0, szx = 0, szy = 0;
    for (i = 0; i < n; i++) { sxx += (xs[i] - mx) * (xs[i] - mx); sxy += (xs[i] - mx) * (ys[i] - my); szx += (zs[i] - mz) * (xs[i] - mx); szy += (zs[i] - mz) * (ys[i] - my); }
    var bo = sxy / sxx, bi = szy / szx, x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var line = function (sl) { return [{ x: x0, y: my + sl * (x0 - mx) }, { x: x1, y: my + sl * (x1 - mx) }]; };
    a.chart('tall', { data: { datasets: [{ type: 'scatter', label: 'データ', data: pts(xs, ys), backgroundColor: c.fill, borderColor: c.c1, pointRadius: 2.2 },
      a.lineDs('真の傾き β = ' + a.fmt(b), line(b), c.fg, { borderDash: [5, 4], borderWidth: 1.5 }),
      a.lineDs('最小二乗法', line(bo), c.c2, { borderWidth: 2.5 }), a.lineDs('操作変数法', line(bi), c.c3, { borderWidth: 2.5 })] },
    options: opts(a, c, { legend: true, ymin: null, xTitle: 'x（内生的な説明変数）', yTitle: 'y' }) });
    a.info.textContent = '最小二乗推定 ' + a.fmt(bo, 3) + '（理論上の偏り ρ/(strength²+1) = ' + a.fmt(rho / (st * st + 1), 3) + '）　操作変数推定 s_zy/s_zx = ' + a.fmt(bi, 3) + '（真値 ' + a.fmt(b) + '）';
  });
  // qq-plot: { dist: norm|t|expon|unif, n, df } 正規Q-Qプロット
  reg('qq-plot', function (a, p, c) {
    var d = p.dist || 'norm', n = Math.round(num(p.n, 50)), df = num(p.df, 3), v = [];
    for (var i = 0; i < n; i++) {
      var u = a.rand();
      v.push(d === 'expon' ? -Math.log(1 - u) : d === 'unif' ? u : d === 't' ? Dist.quantile('t', u, { df: df }) : a.normal());
    }
    var s = sorted(v), qs = s.map(function (_, i) { return Dist.quantile('norm', (i + 0.5) / n, {}); });
    var q1 = quantileSorted(s, 0.25), q3 = quantileSorted(s, 0.75), z1 = Dist.quantile('norm', 0.25, {}), sl = (q3 - q1) / (2 * -z1), ic = q1 - sl * z1;
    var ds = [{ type: 'scatter', label: '標本', data: pts(qs, s), backgroundColor: c.c1, pointRadius: 3 }, a.lineDs('四分位点を通る直線', [{ x: -3, y: ic - 3 * sl }, { x: 3, y: ic + 3 * sl }], c.c2, { borderDash: [6, 4] })];
    a.chart('tall', { data: { datasets: ds }, options: opts(a, c, { legend: true, ymin: null, xmin: -3, xmax: 3, xTitle: '標準正規分布の分位点', yTitle: '標本の分位点（並べた値）' }) });
    a.info.textContent = { norm: '正規分布からの標本：ほぼ直線に並ぶ', t: '裾の重い t 分布：両端が直線から外側へ反る（S字）', expon: '右に歪んだ指数分布：下に凸の弓なり', unif: '裾の軽い一様分布：両端が内側へ曲がる' }[d] || '';
  });

  // ---------- 第19・30章 ----------
  // ridge-lasso-path: { method: ridge|lasso } 4変数のデータで係数の推移（横軸 log10 λ）
  function pathData(seed) {
    var r = Charts._rng(seed), z = Charts._normal(r), n = 60, X = [], y = [], beta = [3, -2, 0, 1.5];
    for (var i = 0; i < n; i++) {
      var f = z(), row = [f + 0.5 * z(), f + 0.5 * z(), z(), z()];
      X.push(row); y.push(row.reduce(function (s, v, j) { return s + v * beta[j]; }, 0) + 2 * z());
    }
    for (var j = 0; j < 4; j++) { var col = X.map(function (r2) { return r2[j]; }), m = mean(col), s = sd(col); X.forEach(function (r2) { r2[j] = (r2[j] - m) / s; }); }
    var my = mean(y); y = y.map(function (v) { return v - my; });
    return { X: X, y: y, n: n };
  }
  reg('ridge-lasso-path', function (a, p, c) {
    var D = pathData(2024), meth = p.method || 'lasso', grid = a.linspace(-2, 2.5, 60), pal = [c.c1, c.c2, c.c3, c.muted], series = [[], [], [], []];
    grid.forEach(function (lg) {
      var lam = Math.pow(10, lg), b;
      if (meth === 'ridge') b = lsq(D.X, D.y, lam);
      else {
        b = [0, 0, 0, 0]; var lamL = lam / 10;
        for (var it = 0; it < 300; it++) for (var j = 0; j < 4; j++) {
          var rho = 0; for (var i = 0; i < D.n; i++) { var r = D.y[i]; for (var k = 0; k < 4; k++) if (k !== j) r -= D.X[i][k] * b[k]; rho += D.X[i][j] * r; }
          rho /= D.n; var zj = 0; for (i = 0; i < D.n; i++) zj += D.X[i][j] * D.X[i][j]; zj /= D.n;
          b[j] = Math.sign(rho) * Math.max(Math.abs(rho) - lamL, 0) / zj;
        }
      }
      b.forEach(function (v, j) { series[j].push({ x: lg, y: v }); });
    });
    var ds = series.map(function (s, j) { return a.lineDs('β' + (j + 1), s, pal[j]); });
    a.chart('', { data: { datasets: ds }, options: opts(a, c, { legend: true, ymin: null, xmin: -2, xmax: 2.5, xTitle: meth === 'ridge' ? 'log10 λ（リッジ）' : 'log10(10λ)（Lasso）', yTitle: '推定された係数' }) });
    a.info.textContent = meth === 'ridge' ? 'リッジ：λ を大きくすると全係数がなめらかに0へ縮むが、ちょうど0にはならない' : 'Lasso：λ を大きくすると係数が順に「ちょうど0」になり、変数選択が起こる（真の値は β = (3, −2, 0, 1.5)）';
  });
  // bias-variance: { degree, noise, n } 多項式の次数と訓練誤差・テスト誤差
  reg('bias-variance', function (a, p, c) {
    var deg = Math.round(num(p.degree, 3)), noise = num(p.noise, 0.3), n = Math.round(num(p.n, 20)), rand = a.rand, z = a.normal;
    var f = function (x) { return Math.sin(Math.PI * x); };
    var design = function (x, d) { var r = []; for (var k = 0; k <= d; k++) r.push(Math.pow(x, k)); return r; };
    var tx = a.linspace(-1, 1, 101), tr = [], te = [], fits = [];
    for (var d = 1; d <= 9; d++) { tr.push(0); te.push(0); }
    for (var rep = 0; rep < 20; rep++) {
      var xs = [], ys = []; for (var i = 0; i < n; i++) { var x = 2 * rand() - 1; xs.push(x); ys.push(f(x) + noise * z()); }
      for (d = 1; d <= 9; d++) {
        var b = lsq(xs.map(function (x) { return design(x, d); }), ys, 1e-9);
        var pr = function (x) { return design(x, d).reduce(function (s, v, k) { return s + v * b[k]; }, 0); };
        tr[d - 1] += mean(xs.map(function (x, i) { return Math.pow(ys[i] - pr(x), 2); })) / 20;
        te[d - 1] += (mean(tx.map(function (x) { return Math.pow(f(x) - pr(x), 2); })) + noise * noise) / 20;
        if (d === deg && rep < 8) fits.push(pts(tx, tx.map(pr)));
      }
    }
    var ds = fits.map(function (fp) { return a.lineDs('', fp, c.fill, { borderWidth: 1, borderColor: c.c3 }); });
    ds.push(a.lineDs('真の関数', pts(tx, tx.map(f)), c.c2, { borderWidth: 2.5 }));
    a.chart('short', { data: { datasets: ds }, options: opts(a, c, { ymin: -2, ymax: 2, xmin: -1, xmax: 1, xTitle: 'x（' + deg + '次多項式を8つのデータセットに当てはめ）' }) });
    var ks = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    a.chart('short', { data: { datasets: [a.lineDs('訓練誤差', pts(ks, tr), c.c1, { pointRadius: 3 }), a.lineDs('テスト誤差（期待予測誤差）', pts(ks, te), c.c2, { pointRadius: 3 })] },
      options: opts(a, c, { legend: true, xmin: 1, xmax: 9, ymax: Math.min(1.5, Math.max.apply(null, te) * 1.1), xTitle: '多項式の次数' }) });
    a.info.textContent = '次数を上げると訓練誤差は下がり続けるが、テスト誤差は途中から上がる（過学習）。最小は ' + (te.indexOf(Math.min.apply(null, te)) + 1) + ' 次';
  });

  // ---------- 第22〜24章 ----------
  // pca-axes: { rho, sx, sy, n } 散布図と主成分の軸
  reg('pca-axes', function (a, p, c) {
    var rho = num(p.rho, 0.8), sx = num(p.sx, 1), sy = num(p.sy, 1), n = Math.round(num(p.n, 200)), xs = [], ys = [];
    for (var i = 0; i < n; i++) { var u = a.normal(), v = a.normal(); xs.push(sx * u); ys.push(sy * (rho * u + Math.sqrt(1 - rho * rho) * v)); }
    var mx = mean(xs), my = mean(ys), sxx = 0, syy = 0, sxy = 0;
    for (i = 0; i < n; i++) { sxx += (xs[i] - mx) * (xs[i] - mx); syy += (ys[i] - my) * (ys[i] - my); sxy += (xs[i] - mx) * (ys[i] - my); }
    sxx /= n - 1; syy /= n - 1; sxy /= n - 1;
    var tr = sxx + syy, det = sxx * syy - sxy * sxy, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(tr * tr / 4 - det);
    var v1 = Math.abs(sxy) > 1e-12 ? [sxy, l1 - sxx] : (sxx >= syy ? [1, 0] : [0, 1]), nv = Math.hypot(v1[0], v1[1]); v1 = [v1[0] / nv, v1[1] / nv];
    var v2 = [-v1[1], v1[0]], k1 = 2 * Math.sqrt(l1), k2 = 2 * Math.sqrt(l2);
    var lim = 3.5 * Math.max(sx, sy);
    var ds = [{ type: 'scatter', label: 'データ', data: pts(xs, ys), backgroundColor: c.fill, borderColor: c.c1, pointRadius: 2.2 },
      a.lineDs('第1主成分', [{ x: mx - k1 * v1[0], y: my - k1 * v1[1] }, { x: mx + k1 * v1[0], y: my + k1 * v1[1] }], c.c2, { borderWidth: 3 }),
      a.lineDs('第2主成分', [{ x: mx - k2 * v2[0], y: my - k2 * v2[1] }, { x: mx + k2 * v2[0], y: my + k2 * v2[1] }], c.c3, { borderWidth: 3 })];
    var o = opts(a, c, { legend: true, ymin: -lim, ymax: lim, xmin: -lim, xmax: lim, xTitle: 'x1', yTitle: 'x2' });
    a.chart('tall', { data: { datasets: ds }, options: o });
    a.info.textContent = '固有値 λ1 = ' + a.fmt(l1, 3) + '，λ2 = ' + a.fmt(l2, 3) + '　第1主成分の寄与率 = ' + a.fmt(l1 / (l1 + l2), 3) + '　固有ベクトル (' + a.fmt(v1[0], 3) + ', ' + a.fmt(v1[1], 3) + ')（線の長さ＝2×標準偏差）';
  });
  function makeBlobs(a, n, k) {
    var centers = [[-2, -1], [2, -1.2], [0, 2.2], [3, 2.5], [-3, 2]].slice(0, k), P = [];
    for (var i = 0; i < n; i++) { var cc = centers[i % k]; P.push([cc[0] + 0.8 * a.normal(), cc[1] + 0.8 * a.normal()]); }
    return P;
  }
  // dendrogram: { n, method: single|complete|average|ward } 散布図とデンドログラム
  reg('dendrogram', function (a, p, c) {
    var n = Math.min(16, Math.round(num(p.n, 10))), meth = p.method || 'ward', P = makeBlobs(a, n, 3);
    var D = P.map(function (u) { return P.map(function (v) { var d2 = Math.pow(u[0] - v[0], 2) + Math.pow(u[1] - v[1], 2); return meth === 'ward' ? d2 : Math.sqrt(d2); }); });
    var active = P.map(function (_, i) { return i; }), size = P.map(function () { return 1; }), node = P.map(function (_, i) { return { leaf: i, h: 0 }; }), merges = [];
    while (active.length > 1) {
      var bi = -1, bj = -1, best = Infinity;
      active.forEach(function (i) { active.forEach(function (j) { if (i < j && D[i][j] < best) { best = D[i][j]; bi = i; bj = j; } }); });
      var ni = size[bi], nj = size[bj];
      active.forEach(function (k) {
        if (k === bi || k === bj) return;
        var nk = size[k], dik = D[bi][k], djk = D[bj][k], v;
        if (meth === 'single') v = Math.min(dik, djk);
        else if (meth === 'complete') v = Math.max(dik, djk);
        else if (meth === 'average') v = (ni * dik + nj * djk) / (ni + nj);
        else v = ((ni + nk) * dik + (nj + nk) * djk - nk * best) / (ni + nj + nk);
        D[bi][k] = D[k][bi] = v;
      });
      node[bi] = { l: node[bi], r: node[bj], h: meth === 'ward' ? Math.sqrt(best) : best };
      size[bi] = ni + nj; merges.push(node[bi].h);
      active = active.filter(function (k) { return k !== bj; });
    }
    var tree = node[active[0]], order = [];
    (function walk(t) { if (t.leaf != null) order.push(t.leaf); else { walk(t.l); walk(t.r); } })(tree);
    var cv = a.canvas(240), ctx = cv.ctx, W = cv.w, H = cv.h, top = 16, bot = H - 26, hmax = tree.h * 1.05;
    var X = function (leaf) { return 24 + (W - 48) * (order.indexOf(leaf) + 0.5) / n; }, Y = function (h) { return bot - (bot - top) * h / hmax; };
    ctx.strokeStyle = c.c1; ctx.lineWidth = 1.6;
    (function draw(t) {
      if (t.leaf != null) return X(t.leaf);
      var xl = draw(t.l), xr = draw(t.r), yl = Y(t.l.h || 0), yr = Y(t.r.h || 0), y = Y(t.h);
      ctx.beginPath(); ctx.moveTo(xl, yl); ctx.lineTo(xl, y); ctx.lineTo(xr, y); ctx.lineTo(xr, yr); ctx.stroke();
      return (xl + xr) / 2;
    })(tree);
    ctx.fillStyle = c.fg; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    order.forEach(function (leaf) { ctx.fillText(String(leaf + 1), X(leaf), bot + 6); });
    var gaps = merges.map(function (h, i) { return i ? h - merges[i - 1] : 0; }), gi = gaps.indexOf(Math.max.apply(null, gaps));
    a.chart('short', { data: { datasets: [{ type: 'scatter', label: '点（番号は下の葉）', data: pts(P.map(function (q) { return q[0]; }), P.map(function (q) { return q[1]; })), backgroundColor: c.c1, pointRadius: 4 }] },
      options: Object.assign(opts(a, c, { ymin: null, xTitle: 'x1', yTitle: 'x2' }), { plugins: { legend: { display: false }, tooltip: { callbacks: { label: function (ctx) { return '点 ' + (ctx.dataIndex + 1); } } } } }) });
    a.info.textContent = { single: '最短距離法', complete: '最長距離法', average: '群平均法', ward: 'ウォード法' }[meth] + '：併合の高さが最も大きく跳ぶのは ' + (n - gi) + ' クラスターから ' + (n - gi - 1) + ' クラスターになるとき';
  });
  // kmeans-steps: { k, n, step } k-means法の反復（step = 0 は初期値）
  reg('kmeans-steps', function (a, p, c) {
    var k = Math.round(num(p.k, 3)), n = Math.round(num(p.n, 90)), step = Math.round(num(p.step, 0)), P = makeBlobs(a, n, 3);
    var C = []; for (var i = 0; i < k; i++) C.push(P[Math.floor(a.rand() * n)].slice());
    var lab = [], assign = function () { lab = P.map(function (q) { var b = 0, bd = Infinity; C.forEach(function (cc, j) { var d = Math.pow(q[0] - cc[0], 2) + Math.pow(q[1] - cc[1], 2); if (d < bd) { bd = d; b = j; } }); return b; }); };
    assign();
    for (var s = 0; s < step; s++) {
      C = C.map(function (cc, j) { var m = P.filter(function (_, i2) { return lab[i2] === j; }); return m.length ? [mean(m.map(function (q) { return q[0]; })), mean(m.map(function (q) { return q[1]; }))] : cc; });
      assign();
    }
    var W = 0; P.forEach(function (q, i2) { W += Math.pow(q[0] - C[lab[i2]][0], 2) + Math.pow(q[1] - C[lab[i2]][1], 2); });
    var pal = [c.c1, c.c2, c.c3, c.muted, c.faint];
    var ds = C.map(function (cc, j) { return { type: 'scatter', label: 'クラスター' + (j + 1), data: P.filter(function (_, i2) { return lab[i2] === j; }).map(function (q) { return { x: q[0], y: q[1] }; }), backgroundColor: pal[j % 5], pointRadius: 3 }; });
    ds.push({ type: 'scatter', label: '中心', data: C.map(function (cc) { return { x: cc[0], y: cc[1] }; }), backgroundColor: c.fg, pointRadius: 9, pointStyle: 'crossRot', borderColor: c.fg, borderWidth: 3 });
    a.chart('tall', { data: { datasets: ds }, options: opts(a, c, { legend: true, ymin: null, xTitle: 'x1', yTitle: 'x2' }) });
    a.info.textContent = '反復 ' + step + ' 回目　クラスター内平方和 W = ' + a.fmt(W, 2) + '（反復ごとに減るか変わらない）';
  });
  // roc-curve: { mu0, mu1, sigma, threshold } 2つの正規分布の判別とROC曲線
  reg('roc-curve', function (a, p, c) {
    var m0 = num(p.mu0, 0), m1 = num(p.mu1, 1.5), sg = num(p.sigma, 1), th = num(p.threshold, 0.75), lo = Math.min(m0, m1) - 4 * sg, hi = Math.max(m0, m1) + 4 * sg, xs = a.linspace(lo, hi, 241);
    var f0 = function (x) { return Dist.pdf('norm', x, { mu: m0, sigma: sg }); }, f1 = function (x) { return Dist.pdf('norm', x, { mu: m1, sigma: sg }); };
    var top = Math.max(f0(m0), f1(m1)) * 1.1;
    a.chart('short', { data: { datasets: [a.lineDs('陰性群', pts(xs, xs.map(f0)), c.c1), a.lineDs('陽性群', pts(xs, xs.map(f1)), c.c2), a.lineDs('しきい値', [{ x: th, y: 0 }, { x: th, y: top }], c.fg, { borderDash: [5, 4], borderWidth: 1 })] },
      options: opts(a, c, { legend: true, xmin: lo, xmax: hi, ymax: top, xTitle: 'スコア（しきい値より大きければ陽性と判定）' }) });
    var ts = a.linspace(hi, lo, 200), roc = ts.map(function (t) { return { x: 1 - Dist.cdf('norm', t, { mu: m0, sigma: sg }), y: 1 - Dist.cdf('norm', t, { mu: m1, sigma: sg }) }; });
    var fpr = 1 - Dist.cdf('norm', th, { mu: m0, sigma: sg }), tpr = 1 - Dist.cdf('norm', th, { mu: m1, sigma: sg }), auc = Dist.special.normCdf((m1 - m0) / (sg * Math.SQRT2));
    a.chart('', { data: { datasets: [a.lineDs('ROC曲線', roc, c.c1, { fill: 'origin', backgroundColor: c.fill }), a.lineDs('ランダム', [{ x: 0, y: 0 }, { x: 1, y: 1 }], c.faint, { borderDash: [4, 4], borderWidth: 1 }),
      { type: 'scatter', label: '現在のしきい値', data: [{ x: fpr, y: tpr }], backgroundColor: c.c2, pointRadius: 6 }] },
    options: opts(a, c, { legend: true, xmin: 0, xmax: 1, ymax: 1, xTitle: '偽陽性率 1 − 特異度', yTitle: '真陽性率（感度）' }) });
    a.info.textContent = '感度 = ' + a.fmt(tpr) + '，特異度 = ' + a.fmt(1 - fpr) + '　AUC = Φ(Δ/(σ√2)) = ' + a.fmt(auc);
  });

  // ---------- 第27章 ----------
  function simArma(a, n, phi, phi2, theta) {
    var x = [], e = [], burn = 100;
    for (var t = 0; t < n + burn; t++) {
      var et = a.normal(), v = et + theta * (e[t - 1] || 0) + phi * (x[t - 1] || 0) + phi2 * (x[t - 2] || 0);
      e.push(et); x.push(v);
    }
    return x.slice(burn);
  }
  function acf(x, L) {
    var n = x.length, m = mean(x), c0 = 0, r = [];
    x.forEach(function (v) { c0 += (v - m) * (v - m); });
    for (var h = 0; h <= L; h++) { var s = 0; for (var t = h; t < n; t++) s += (x[t] - m) * (x[t - h] - m); r.push(s / c0); }
    return r;
  }
  function pacfFromAcf(r, L) {
    var out = [1], phiPrev = [];
    for (var k = 1; k <= L; k++) {
      var num2 = r[k], den = 1, j;
      for (j = 1; j < k; j++) { num2 -= phiPrev[j - 1] * r[k - j]; den -= phiPrev[j - 1] * r[j]; }
      var pkk = num2 / den, cur = [];
      for (j = 1; j < k; j++) cur.push(phiPrev[j - 1] - pkk * phiPrev[k - j - 1]);
      cur.push(pkk); phiPrev = cur; out.push(pkk);
    }
    return out;
  }
  // acf-plot: { model: ar1|ar2|ma1|white, phi, phi2, theta, n, lags, pacf: 1 で偏自己相関も }
  reg('acf-plot', function (a, p, c) {
    var model = p.model || 'ar1', phi = num(p.phi, 0.7), phi2 = num(p.phi2, 0), th = num(p.theta, 0.6), n = Math.round(num(p.n, 200)), L = Math.round(num(p.lags, 20));
    var x = simArma(a, n, model === 'ar1' || model === 'ar2' ? phi : 0, model === 'ar2' ? phi2 : 0, model === 'ma1' ? th : 0);
    var r = acf(x, L), band = 1.96 / Math.sqrt(n), lags = []; for (var h = 0; h <= L; h++) lags.push(h);
    var theo = lags.map(function (h) {
      if (model === 'ar1') return Math.pow(phi, h);
      if (model === 'ma1') return h === 0 ? 1 : h === 1 ? th / (1 + th * th) : 0;
      if (model === 'white') return h === 0 ? 1 : 0;
      return null;
    });
    if (model === 'ar2') { var rho = [1, phi / (1 - phi2)]; for (h = 2; h <= L; h++) rho.push(phi * rho[h - 1] + phi2 * rho[h - 2]); theo = rho; }
    a.chart('short', { data: { datasets: [a.lineDs('時系列', pts(x.map(function (_, t) { return t + 1; }), x), c.c1, { borderWidth: 1 })] }, options: opts(a, c, { ymin: null, xmin: 1, xmax: n, xTitle: '時点 t' }) });
    var mk = function (vals, label, th2) {
      return [barDs(label, vals, c, { barPercentage: 0.4 }),
        a.lineDs('±1.96/√n', lags.map(function () { return band; }), c.faint, { borderDash: [4, 4], borderWidth: 1 }),
        a.lineDs('', lags.map(function () { return -band; }), c.faint, { borderDash: [4, 4], borderWidth: 1 })].concat(th2 ? [a.lineDs('理論値', th2, c.c2, { showLine: false, pointRadius: 3 })] : []);
    };
    a.chart('short', { data: { labels: lags, datasets: mk(r, '標本自己相関', theo) }, options: opts(a, c, { xType: 'category', legend: true, ymin: -1, ymax: 1, xTitle: 'ラグ h' }) });
    if (p.pacf) {
      var pr = pacfFromAcf(r, L);
      a.chart('short', { data: { labels: lags, datasets: mk(pr, '標本偏自己相関') }, options: opts(a, c, { xType: 'category', legend: true, ymin: -1, ymax: 1, xTitle: 'ラグ h' }) });
    }
    a.info.textContent = { ar1: 'AR(1)：自己相関は φ^h で指数的に減衰、偏自己相関はラグ1の後で切れる', ar2: 'AR(2)：偏自己相関はラグ2の後で切れる', ma1: 'MA(1)：自己相関はラグ1の後で切れ、偏自己相関はゆっくり減衰', white: 'ホワイトノイズ：ラグ1以降の自己相関はほぼ帯の中' }[model] || '';
  });
  // ar-sim: { phi, phi2, n } AR(1)/AR(2) の系列
  reg('ar-sim', function (a, p, c) {
    var phi = num(p.phi, 0.7), phi2 = num(p.phi2, 0), n = Math.round(num(p.n, 200));
    var x = [0, 0]; for (var t = 2; t < n + 2; t++) x.push(phi * x[t - 1] + phi2 * x[t - 2] + a.normal());
    x = x.slice(2);
    var stat = phi2 === 0 ? Math.abs(phi) < 1 : (phi + phi2 < 1 && phi2 - phi < 1 && Math.abs(phi2) < 1);
    a.chart('', { data: { datasets: [a.lineDs('X_t', pts(x.map(function (_, i) { return i + 1; }), x), c.c1, { borderWidth: 1.2 })] }, options: opts(a, c, { ymin: null, xmin: 1, xmax: n, xTitle: '時点 t' }) });
    a.info.textContent = 'φ1 = ' + a.fmt(phi) + (phi2 ? '，φ2 = ' + a.fmt(phi2) : '') + '：' + (stat ? '定常（平均0のまわりを行き来する）' : '非定常（発散や単位根のふるまい）') +
      (phi2 === 0 && stat ? '　分散 1/(1−φ²) = ' + a.fmt(1 / (1 - phi * phi)) : '');
  });

  // ---------- 第28〜29章 ----------
  // mosaic: { table: [[...],[...]], rowLabels, colLabels } モザイク図、オッズ比、カイ二乗統計量
  reg('mosaic', function (a, p, c) {
    var T = p.table || [[30, 10], [20, 40]], rl = p.rowLabels || T.map(function (_, i) { return '行' + (i + 1); }), cl = p.colLabels || T[0].map(function (_, j) { return '列' + (j + 1); });
    var R = T.length, C = T[0].length, rs = T.map(function (r) { return r.reduce(function (s, v) { return s + v; }, 0); }), N = rs.reduce(function (s, v) { return s + v; }, 0);
    var cs = cl.map(function (_, j) { return T.reduce(function (s, r) { return s + r[j]; }, 0); });
    var cv = a.canvas(240), ctx = cv.ctx, W = cv.w, H = cv.h, x0 = 60, y0 = 24, w = W - x0 - 16, h = H - y0 - 30, pal = [c.c1, c.c2, c.c3, c.muted, c.faint], gap = 3;
    var x = x0;
    T.forEach(function (row, i) {
      var wi = w * rs[i] / N - gap, y = y0;
      row.forEach(function (v, j) {
        var hj = h * v / rs[i];
        ctx.fillStyle = pal[j % 5]; ctx.globalAlpha = 0.75; ctx.fillRect(x, y, Math.max(1, wi), Math.max(0, hj - gap)); ctx.globalAlpha = 1;
        if (hj > 16 && wi > 30) { ctx.fillStyle = c.surface; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(v), x + wi / 2, y + hj / 2); }
        y += hj;
      });
      ctx.fillStyle = c.fg; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(rl[i], x + wi / 2, y0 + h + 6);
      x += wi + gap;
    });
    ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    var lx = x0; cl.forEach(function (lab, j) { ctx.fillStyle = pal[j % 5]; ctx.fillRect(lx, 6, 12, 12); ctx.fillStyle = c.fg; ctx.fillText(lab, lx + 16, 19); lx += ctx.measureText(lab).width + 34; });
    var chi = 0; T.forEach(function (r, i) { r.forEach(function (v, j) { var e = rs[i] * cs[j] / N; chi += (v - e) * (v - e) / e; }); });
    var df = (R - 1) * (C - 1), pv = 1 - Dist.cdf('chi2', chi, { df: df });
    var info = 'カイ二乗統計量 = ' + a.fmt(chi, 3) + '（自由度 ' + df + '，p値 = ' + a.fmt(pv) + '）';
    if (R === 2 && C === 2) info = 'オッズ比 = (' + T[0][0] + '×' + T[1][1] + ')/(' + T[0][1] + '×' + T[1][0] + ') = ' + a.fmt(T[0][0] * T[1][1] / (T[0][1] * T[1][0]), 3) + '　' + info;
    a.info.textContent = info + '　横幅＝行の割合、縦＝行の中での列の割合';
  });
  // km-curve: { n, rate, censor, groups: 1|2, hr } カプラン・マイヤー曲線（2群ならログランク検定も）
  reg('km-curve', function (a, p, c) {
    var n = Math.round(num(p.n, 30)), rate = num(p.rate, 0.1), cr = num(p.censor, 0.04), G = Math.round(num(p.groups, 1)), hr = num(p.hr, 2);
    var data = [];
    for (var g = 0; g < G; g++) for (var i = 0; i < n; i++) { var t = expDraw(a.rand, rate * (g ? hr : 1)), cz = cr > 0 ? expDraw(a.rand, cr) : Infinity; data.push({ g: g, t: Math.min(t, cz), d: t <= cz ? 1 : 0 }); }
    var pal = [c.c1, c.c2], ds = [], medians = [], tmax = Math.max.apply(null, data.map(function (r) { return r.t; }));
    for (g = 0; g < G; g++) {
      var rows = data.filter(function (r) { return r.g === g; }).sort(function (x, y) { return x.t - y.t; }), S = 1, atRisk = rows.length, line = [{ x: 0, y: 1 }], cens = [], med = null;
      rows.forEach(function (r) {
        if (r.d) { line.push({ x: r.t, y: S }); S *= 1 - 1 / atRisk; line.push({ x: r.t, y: S }); if (med == null && S <= 0.5) med = r.t; }
        else cens.push({ x: r.t, y: S });
        atRisk--;
      });
      line.push({ x: tmax, y: S });
      ds.push(a.lineDs(G > 1 ? '群' + (g + 1) : 'KM推定', line, pal[g]));
      ds.push({ type: 'scatter', label: '打ち切り', data: cens, borderColor: pal[g], backgroundColor: pal[g], pointStyle: 'line', pointRadius: 6, rotation: 90, borderWidth: 2 });
      medians.push(med);
    }
    a.chart('', { data: { datasets: ds }, options: opts(a, c, { legend: true, xmin: 0, xmax: tmax, ymax: 1.02, xTitle: '時間', yTitle: '生存確率 Ŝ(t)' }) });
    var info = '中央生存時間：' + medians.map(function (m, k) { return (G > 1 ? '群' + (k + 1) + ' ' : '') + (m == null ? '未到達' : a.fmt(m, 3)); }).join('，');
    if (G === 2) {
      var times = sorted(data.filter(function (r) { return r.d; }).map(function (r) { return r.t; })), O1 = 0, E1 = 0, V = 0;
      times.forEach(function (t2, k) {
        if (k && t2 === times[k - 1]) return;
        var r1 = data.filter(function (r) { return r.g === 0 && r.t >= t2; }).length, r2 = data.filter(function (r) { return r.g === 1 && r.t >= t2; }).length;
        var d1 = data.filter(function (r) { return r.g === 0 && r.d && r.t === t2; }).length, d = data.filter(function (r) { return r.d && r.t === t2; }).length, rr = r1 + r2;
        O1 += d1; E1 += d * r1 / rr; if (rr > 1) V += d * (r1 / rr) * (r2 / rr) * (rr - d) / (rr - 1);
      });
      var chi = (O1 - E1) * (O1 - E1) / V;
      info += '　ログランク統計量 = ' + a.fmt(chi, 3) + '（p値 ' + a.fmt(1 - Dist.cdf('chi2', chi, { df: 1 })) + '）';
    }
    a.info.textContent = info + '　縦線の印は打ち切り';
  });

  // ---------- 第31章 ----------
  // prior-posterior: model: beta-binomial { a, b, n, x } / gamma-poisson { shape, rate, n, sum } / normal-normal { mu0, tau0, sigma, n, xbar }
  reg('prior-posterior', function (a, p, c) {
    var model = p.model || 'beta-binomial', prior, post, lik, lo, hi, dn, pp, qp, info = '';
    if (model === 'gamma-poisson') {
      var sh = num(p.shape, 2), rt = num(p.rate, 1), n = Math.round(num(p.n, 5)), sm = num(p.sum, 12);
      dn = 'gamma'; qp = { shape: sh, scale: 1 / rt }; pp = { shape: sh + sm, scale: 1 / (rt + n) };
      lik = function (l) { return l > 0 ? Math.exp(sm * Math.log(l) - n * l) : 0; };
      hi = Math.max(Dist.quantile('gamma', 0.995, qp), Dist.quantile('gamma', 0.999, pp)); lo = 0;
      info = '事後分布 Ga(' + a.fmt(sh + sm) + ', レート ' + a.fmt(rt + n) + ')　事後平均 ' + a.fmt((sh + sm) / (rt + n));
    } else if (model === 'normal-normal') {
      var m0 = num(p.mu0, 0), t0 = num(p.tau0, 2), sg = num(p.sigma, 1), nn = Math.round(num(p.n, 5)), xb = num(p.xbar, 1.5);
      var prec = 1 / (t0 * t0) + nn / (sg * sg), mpost = (m0 / (t0 * t0) + nn * xb / (sg * sg)) / prec;
      dn = 'norm'; qp = { mu: m0, sigma: t0 }; pp = { mu: mpost, sigma: Math.sqrt(1 / prec) };
      lik = function (m) { return Math.exp(-nn * (xb - m) * (xb - m) / (2 * sg * sg)); };
      lo = Math.min(m0 - 3 * t0, xb - 4 * sg / Math.sqrt(nn)); hi = Math.max(m0 + 3 * t0, xb + 4 * sg / Math.sqrt(nn));
      info = '事後分布 N(' + a.fmt(mpost) + ', ' + a.fmt(1 / prec) + ')　重み：データ ' + a.fmt((nn / (sg * sg)) / prec) + '，事前 ' + a.fmt((1 / (t0 * t0)) / prec);
    } else {
      var aa = num(p.a, 2), bb = num(p.b, 2), N = Math.round(num(p.n, 10)), x = Math.min(N, Math.round(num(p.x, 7)));
      dn = 'beta'; qp = { a: aa, b: bb }; pp = { a: aa + x, b: bb + N - x };
      lik = function (t) { return t > 0 && t < 1 ? Math.exp(x * Math.log(t) + (N - x) * Math.log(1 - t)) : (t === 0 && x === 0) || (t === 1 && x === N) ? 1 : 0; };
      lo = 0; hi = 1;
      info = '事後分布 Be(' + a.fmt(aa + x) + ', ' + a.fmt(bb + N - x) + ')　事後平均 ' + a.fmt((aa + x) / (aa + bb + N)) + (aa + x > 1 && bb + N - x > 1 ? '，MAP ' + a.fmt((aa + x - 1) / (aa + bb + N - 2)) : '');
    }
    var xs = a.linspace(lo, hi, 301), lv = xs.map(lik), area = 0;
    for (var i = 1; i < xs.length; i++) area += (lv[i] + lv[i - 1]) / 2 * (xs[i] - xs[i - 1]);
    var ci = [Dist.quantile(dn, 0.025, pp), Dist.quantile(dn, 0.975, pp)];
    var post2 = xs.map(function (t) { return Dist.pdf(dn, t, pp); }), prior2 = xs.map(function (t) { return Dist.pdf(dn, t, qp); });
    var cap = Math.max.apply(null, post2.filter(isFinite)) * 1.15;
    var ds = [a.lineDs('事前分布', pts(xs, prior2), c.muted, { borderDash: [6, 4] }), a.lineDs('尤度（面積1に調整）', pts(xs, lv.map(function (v) { return v / area; })), c.c2),
      a.lineDs('事後分布', pts(xs, post2), c.c1, { borderWidth: 3 }),
      a.lineDs('95%信用区間', pts(xs, xs.map(function (t, k) { return t >= ci[0] && t <= ci[1] ? post2[k] : null; })), c.c1, { fill: 'origin', backgroundColor: c.fill, borderWidth: 0 })];
    a.chart('', { data: { datasets: ds }, options: opts(a, c, { legend: true, xmin: lo, xmax: hi, ymax: cap, xTitle: 'パラメータ' }) });
    a.info.textContent = info + '　95%信用区間 [' + a.fmt(ci[0]) + ', ' + a.fmt(ci[1]) + ']';
  });

  // ---------- 第32章 ----------
  // mcmc-trace: { target: norm|normmix, step, iters, burnin } ランダムウォーク型メトロポリス法
  reg('mcmc-trace', function (a, p, c) {
    var tgt = p.target || 'normmix', st = num(p.step, 1), N = Math.round(num(p.iters, 3000)), B = Math.round(num(p.burnin, 300));
    var tp = tgt === 'norm' ? { mu: 0, sigma: 1 } : { w: 0.3, mu1: -2, sigma1: 0.6, mu2: 2, sigma2: 1 }, dn = tgt === 'norm' ? 'norm' : 'normmix';
    var f = function (x) { return Dist.pdf(dn, x, tp); }, x = num(p.init, -6), acc = 0, chain = [];
    for (var i = 0; i < N; i++) { var y = x + st * a.normal(); if (a.rand() < f(y) / f(x)) { x = y; acc++; } chain.push(x); }
    a.chart('short', { data: { datasets: [a.lineDs('連鎖の値', chain.map(function (v, t) { return { x: t + 1, y: v }; }), c.c1, { borderWidth: 1 }),
      a.lineDs('バーンイン', [{ x: B, y: -6 }, { x: B, y: 6 }], c.c2, { borderDash: [5, 4], borderWidth: 1.5 })] },
    options: opts(a, c, { legend: true, ymin: -6, ymax: 6, xmin: 1, xmax: N, xTitle: '反復' }) });
    var xs = a.linspace(-6, 6, 241);
    a.chart('short', { data: { datasets: [barDs('バーンイン後の標本', a.hist(chain.slice(B), 48, -6, 6), c), a.lineDs('目標分布', pts(xs, xs.map(f)), c.c2)] },
      options: opts(a, c, { legend: true, xmin: -6, xmax: 6, xTitle: 'x' }) });
    a.info.textContent = '提案の幅 ' + a.fmt(st) + '　採択率 = ' + a.fmt(acc / N, 3) + '（幅が小さすぎると採択率は高いが動きが遅く、大きすぎると採択されにくい）';
  });
  // inverse-transform: { dist, n, ...その分布のパラメータ } 逆関数法
  reg('inverse-transform', function (a, p, c) {
    var dn = p.dist || 'expon', q = distParams(dn, p), n = Math.round(num(p.n, 1000)), r = Dist.range(dn, q), xs = a.linspace(r[0], r[1], 241);
    var us = [], vs = []; for (var i = 0; i < n; i++) { var u = a.rand(); us.push(u); vs.push(Dist.quantile(dn, u, q)); }
    var ds = [a.lineDs('F(x)', pts(xs, xs.map(function (x) { return Dist.cdf(dn, x, q); })), c.c1)];
    for (i = 0; i < 6; i++) ds.push(a.lineDs(i ? '' : 'u → x = F⁻¹(u)', [{ x: r[0], y: us[i] }, { x: vs[i], y: us[i] }, { x: vs[i], y: 0 }], c.c2, { borderDash: [4, 3], borderWidth: 1, pointRadius: [0, 3, 0] }));
    a.chart('short', { data: { datasets: ds }, options: opts(a, c, { legend: true, xmin: r[0], xmax: r[1], ymax: 1, xTitle: 'x' }) });
    a.chart('short', { data: { datasets: [barDs('生成した ' + n + ' 個', a.hist(vs, 40, r[0], r[1]), c), a.lineDs('密度 f(x)', pts(xs, xs.map(function (x) { return Dist.pdf(dn, x, q); })), c.c2)] },
      options: opts(a, c, { legend: true, xmin: r[0], xmax: r[1], xTitle: 'x' }) });
    a.info.textContent = '一様乱数 u を縦軸にとり、累積分布関数の逆関数で横軸の値に写す　生成値の平均 = ' + a.fmt(mean(vs));
  });
  // rejection-sampling: { a, b, n } ベータ分布を一様分布の提案から棄却法で生成
  reg('rejection-sampling', function (a, p, c) {
    var q = { a: num(p.a, 2), b: num(p.b, 5) }, n = Math.round(num(p.n, 600)), xs = a.linspace(0.001, 0.999, 241);
    var M = Math.max.apply(null, xs.map(function (x) { return Dist.pdf('beta', x, q); })) * 1.02, acc = [], rej = [];
    for (var i = 0; i < n; i++) { var x = a.rand(), y = a.rand() * M; (y <= Dist.pdf('beta', x, q) ? acc : rej).push({ x: x, y: y }); }
    a.chart('tall', { data: { datasets: [{ type: 'scatter', label: '採択', data: acc, backgroundColor: c.c1, pointRadius: 2.2 }, { type: 'scatter', label: '棄却', data: rej, backgroundColor: c.faint, pointRadius: 2 },
      a.lineDs('目標 f(x)', pts(xs, xs.map(function (x) { return Dist.pdf('beta', x, q); })), c.c2, { borderWidth: 2.5 }), a.lineDs('包絡 M·g(x)', [{ x: 0, y: M }, { x: 1, y: M }], c.fg, { borderDash: [5, 4], borderWidth: 1 })] },
    options: opts(a, c, { legend: true, xmin: 0, xmax: 1, ymax: M * 1.05, xTitle: 'x' }) });
    a.info.textContent = 'M = ' + a.fmt(M, 3) + '　採択率 = ' + a.fmt(acc.length / n, 3) + '（理論値 1/M = ' + a.fmt(1 / M, 3) + '）';
  });
  // bootstrap-hist: { n, B, stat: mean|median } 指数分布の標本からのブートストラップ
  reg('bootstrap-hist', function (a, p, c) {
    var n = Math.round(num(p.n, 20)), B = Math.round(num(p.B, 1000)), stat = p.stat === 'median' ? function (v) { return quantileSorted(sorted(v), 0.5); } : mean;
    var x = []; for (var i = 0; i < n; i++) x.push(expDraw(a.rand, 1));
    var t0 = stat(x), bs = [];
    for (var b = 0; b < B; b++) { var r = []; for (i = 0; i < n; i++) r.push(x[Math.floor(a.rand() * n)]); bs.push(stat(r)); }
    var s = sorted(bs), lo = quantileSorted(s, 0.025), hi = quantileSorted(s, 0.975), mn = s[0], mx = s[s.length - 1];
    var h = a.hist(bs, 36, mn, mx + 1e-9), top = Math.max.apply(null, h.map(function (q) { return q.y; })) * 1.1;
    a.chart('', { data: { datasets: [barDs('ブートストラップ統計量', h, c), a.lineDs('元の標本の値', [{ x: t0, y: 0 }, { x: t0, y: top }], c.c2, { borderWidth: 2 }),
      a.lineDs('2.5%・97.5%点', [{ x: lo, y: 0 }, { x: lo, y: top }, { x: NaN, y: NaN }, { x: hi, y: top }, { x: hi, y: 0 }], c.fg, { borderDash: [5, 4], borderWidth: 1 })] },
    options: opts(a, c, { legend: true, xmin: mn, xmax: mx, ymax: top, xTitle: p.stat === 'median' ? '標本中央値' : '標本平均' }) });
    a.info.textContent = '元の値 ' + a.fmt(t0) + '　ブートストラップ標準誤差 ' + a.fmt(sd(bs)) + '　95%パーセンタイル区間 [' + a.fmt(lo) + ', ' + a.fmt(hi) + ']（B = ' + B + '）';
  });
})(typeof globalThis !== 'undefined' ? globalThis : this);
