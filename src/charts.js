/* グラフの描画（Chart.js と Canvas）。用語の charts 指定から図を作る */
(function (root) {
  'use strict';
  var Dist = root.Dist;
  var live = [];       // 表示中の図
  var customs = {};    // カスタム図の登録先
  var randomIds = {};  // 乱数を使うカスタム図（「もう一度」ボタンを付ける）

  // ---------- 共通の道具 ----------
  function colors() {
    var s = getComputedStyle(document.documentElement);
    function v(n) { return s.getPropertyValue(n).trim(); }
    return { c1: v('--chart-1'), c2: v('--chart-2'), c3: v('--chart-3'), fill: v('--chart-fill'), fill2: v('--chart-fill-2'),
      grid: v('--chart-grid'), fg: v('--fg'), muted: v('--muted'), faint: v('--faint'), bg: v('--bg'), line: v('--line'), surface: v('--surface'), font: v('--font') };
  }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function fmt(x, d) {
    if (x == null || !isFinite(x)) return '—';
    d = d == null ? 4 : d;
    var a = Math.abs(x);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return x.toExponential(2);
    return String(Number(x.toFixed(d)));
  }
  function linspace(a, b, n) { var r = [], i; for (i = 0; i < n; i++) r.push(a + (b - a) * i / (n - 1)); return r; }
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normalRng(rand) {
    var spare = null;
    return function () {
      if (spare !== null) { var s = spare; spare = null; return s; }
      var u = 0, v = 0;
      while (u === 0) u = rand();
      v = rand();
      var r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
  }
  // 発散する密度（ガンマ形状<1など）でも軸が潰れないよう、y の上限を決める
  function robustMax(vals) {
    var f = vals.filter(function (v) { return v != null && isFinite(v); }).sort(function (a, b) { return a - b; });
    if (!f.length) return undefined;
    var top = f[f.length - 1], q = f[Math.floor(0.94 * (f.length - 1))];
    return top > 4 * q && q > 0 ? q * 2.2 : undefined;
  }
  function cleanVals(vals, cap) {
    return vals.map(function (v) { return (v == null || !isFinite(v)) ? (cap ? cap : null) : (cap && v > cap ? cap : v); });
  }

  // ヒストグラム（密度に換算）。戻り値は Chart.js の {x, y} 点の配列
  function hist(vals, bins, lo, hi) {
    var w = (hi - lo) / bins, cnt = new Array(bins).fill(0), n = 0;
    vals.forEach(function (v) { var k = Math.floor((v - lo) / w); if (k >= 0 && k < bins) cnt[k]++; n++; });
    return cnt.map(function (c, k) { return { x: lo + (k + 0.5) * w, y: c / (n * w) }; });
  }

  // 閉じた形の期待値と分散（存在しなければ null）
  function moments(dist, q) {
    var G = function (x) { return Math.exp(Dist.special.lgamma(x)); };
    switch (dist) {
      case 'unif': return [(q.a + q.b) / 2, Math.pow(q.b - q.a, 2) / 12];
      case 'norm': return [q.mu, q.sigma * q.sigma];
      case 'lognorm': return [Math.exp(q.mu + q.sigma * q.sigma / 2), (Math.exp(q.sigma * q.sigma) - 1) * Math.exp(2 * q.mu + q.sigma * q.sigma)];
      case 'expon': return [1 / q.rate, 1 / (q.rate * q.rate)];
      case 'gamma': return [q.shape * q.scale, q.shape * q.scale * q.scale];
      case 'beta': var s = q.a + q.b; return [q.a / s, q.a * q.b / (s * s * (s + 1))];
      case 'cauchy': return [null, null];
      case 'weibull': var g1 = G(1 + 1 / q.shape); return [q.scale * g1, q.scale * q.scale * (G(1 + 2 / q.shape) - g1 * g1)];
      case 'logistic': return [q.loc, q.scale * q.scale * Math.PI * Math.PI / 3];
      case 'chi2': return [q.df, 2 * q.df];
      case 't': return [q.df > 1 ? 0 : null, q.df > 2 ? q.df / (q.df - 2) : null];
      case 'f': return [q.df2 > 2 ? q.df2 / (q.df2 - 2) : null,
        q.df2 > 4 ? 2 * q.df2 * q.df2 * (q.df1 + q.df2 - 2) / (q.df1 * Math.pow(q.df2 - 2, 2) * (q.df2 - 4)) : null];
      case 'ncx2': return [q.df + q.nc, 2 * (q.df + 2 * q.nc)];
      case 'nct':
        if (q.df <= 1) return [null, null];
        var m = q.nc * Math.sqrt(q.df / 2) * Math.exp(Dist.special.lgamma((q.df - 1) / 2) - Dist.special.lgamma(q.df / 2));
        return [m, q.df > 2 ? q.df * (1 + q.nc * q.nc) / (q.df - 2) - m * m : null];
      case 'ncf':
        var d1 = q.df1, d2 = q.df2, l = q.nc;
        return [d2 > 2 ? d2 * (d1 + l) / (d1 * (d2 - 2)) : null,
          d2 > 4 ? 2 * Math.pow(d2 / d1, 2) * (Math.pow(d1 + l, 2) + (d1 + 2 * l) * (d2 - 2)) / (Math.pow(d2 - 2, 2) * (d2 - 4)) : null];
      case 'normmix':
        var mm = q.w * q.mu1 + (1 - q.w) * q.mu2;
        return [mm, q.w * (q.sigma1 * q.sigma1 + q.mu1 * q.mu1) + (1 - q.w) * (q.sigma2 * q.sigma2 + q.mu2 * q.mu2) - mm * mm];
      default:
        var mo = Dist.moments(dist, q); return [mo.mean, mo.variance];
    }
  }
  function momentText(dist, q) {
    var m = moments(dist, Dist.withDefaults(dist, q));
    return '期待値 ' + (m[0] == null ? '存在しない' : fmt(m[0])) + '　分散 ' + (m[1] == null ? '存在しない' : fmt(m[1]));
  }

  function hasChartJs() { return typeof root.Chart === 'function'; }

  function baseOptions(c, opt) {
    opt = opt || {};
    var tick = { color: c.muted, font: { size: 11 } };
    var o = {
      animation: false, responsive: true, maintainAspectRatio: false, parsing: opt.parsing,
      interaction: { mode: 'nearest', intersect: false, axis: 'x' },
      plugins: {
        legend: { display: !!opt.legend, labels: { color: c.fg, boxWidth: 14, font: { size: 11 } } },
        tooltip: { callbacks: { label: function (ctx) { var y = ctx.parsed.y; return (ctx.dataset.label ? ctx.dataset.label + ': ' : '') + fmt(y); } } }
      },
      scales: {
        x: { type: opt.xType || 'linear', min: opt.xmin, max: opt.xmax, grid: { color: c.grid }, ticks: tick, border: { color: c.line },
          title: { display: !!opt.xTitle, text: opt.xTitle, color: c.muted } },
        y: { min: opt.ymin === undefined ? 0 : (opt.ymin === null ? undefined : opt.ymin), max: opt.ymax, grid: { color: c.grid }, ticks: tick, border: { color: c.line },
          title: { display: !!opt.yTitle, text: opt.yTitle, color: c.muted } }
      }
    };
    if (opt.xType === 'linear') o.scales.x.ticks = Object.assign({}, tick, { maxTicksLimit: 9 });
    return o;
  }
  function lineDs(label, pts, color, extra) {
    return Object.assign({ type: 'line', label: label, data: pts, borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 0, tension: 0, spanGaps: false }, extra || {});
  }

  // 1つの図の枠組み（キャンバス・スライダー・説明）
  function Frame(host, spec) {
    this.spec = spec;
    this.params = Object.assign({}, spec.params || {});
    this.seed = spec.seed || 12345;
    this.charts = [];
    this.fig = el('figure', 'chart');
    this.plot = el('div', 'chart-plot');
    this.info = el('div', 'chart-info');
    this.controls = el('div', 'chart-controls');
    this.fig.appendChild(this.plot);
    this.fig.appendChild(this.info);
    this.fig.appendChild(this.controls);
    if (spec.caption) this.fig.appendChild(el('figcaption', 'chart-cap', spec.caption));
    host.appendChild(this.fig);
  }
  Frame.prototype.canvas = function (cls) {
    var wrap = el('div', 'chart-canvas' + (cls ? ' ' + cls : ''));
    var cv = document.createElement('canvas');
    wrap.appendChild(cv);
    this.plot.appendChild(wrap);
    return cv;
  };
  Frame.prototype.chart = function (cv, config) {
    var ch = new root.Chart(cv.getContext('2d'), config);
    this.charts.push(ch);
    return ch;
  };
  Frame.prototype.clear = function () {
    this.charts.forEach(function (c) { c.destroy(); });
    this.charts = [];
    this.plot.innerHTML = '';
    this.info.textContent = '';
  };
  Frame.prototype.slider = function (s, get, set, onChange) {
    var row = el('div', 'slider');
    var id = 'sl-' + Math.random().toString(36).slice(2, 9);
    var lab = el('label', null, s.label || s.param);
    lab.htmlFor = id;
    var inp = document.createElement('input');
    inp.type = 'range'; inp.id = id;
    inp.min = s.min; inp.max = s.max; inp.step = s.step || 'any';
    inp.value = get();
    var out = el('output', null, fmt(Number(get())));
    inp.addEventListener('input', function () {
      set(Number(inp.value));
      out.textContent = fmt(Number(inp.value));
      onChange();
    });
    row.appendChild(lab); row.appendChild(inp); row.appendChild(out);
    this.controls.appendChild(row);
  };
  Frame.prototype.paramSliders = function (onChange) {
    var self = this;
    (this.spec.sliders || []).forEach(function (s) {
      if (self.params[s.param] == null) self.params[s.param] = s.min;
      self.slider(s, function () { return self.params[s.param]; }, function (v) { self.params[s.param] = v; }, onChange);
    });
  };
  Frame.prototype.button = function (label, fn) {
    var b = el('button', 'chart-btn', label);
    b.type = 'button';
    b.addEventListener('click', fn);
    this.controls.appendChild(b);
  };
  Frame.prototype.missing = function (msg) {
    this.plot.innerHTML = '';
    this.plot.appendChild(el('div', 'chart-missing', msg));
  };

  // 軸の端をきりのよい値にする
  function nice(v, up) {
    if (v === 0 || !isFinite(v)) return v;
    var e = Math.pow(10, Math.floor(Math.log10(Math.abs(v))) - 1), x = v / e;
    return (up ? Math.ceil(x) : Math.floor(x)) * e;
  }
  function xRange(spec, dist, params) {
    var r = Dist.range(dist, params);
    if (!Dist.isDiscrete(dist)) r = [nice(r[0], false), nice(r[1], true)];
    var x = spec.x || {};
    return [x.min != null ? x.min : r[0], x.max != null ? x.max : r[1]];
  }

  // ---------- 各タイプ ----------
  var types = {};

  types.pmf = function (f) {
    var s = f.spec, c = colors(), d = s.dist, q = f.params, r = xRange(s, d, q);
    var lo = Math.ceil(r[0]), hi = Math.floor(r[1]), ks = [], ps = [];
    for (var k = lo; k <= hi; k++) { ks.push(k); ps.push(Dist.pmf(d, k, q)); }
    var ds = [{ type: 'bar', label: 'P(X=k)', data: ps, backgroundColor: c.fill, borderColor: c.c1, borderWidth: 1.5, barPercentage: 0.9, categoryPercentage: 0.95 }];
    var ov = s.overlay;
    var m = moments(d, Dist.withDefaults(d, q));
    if (ov === 'norm-approx' && m[1] > 0) {
      ds.push(lineDs('正規近似', ks.map(function (k) { return Dist.pdf('norm', k, { mu: m[0], sigma: Math.sqrt(m[1]) }); }), c.c2, { tension: 0.35, borderDash: [6, 4] }));
    } else if (ov === 'poisson-approx') {
      var lam = m[0];
      ds.push(lineDs('ポアソン近似', ks.map(function (k) { return Dist.pmf('poisson', k, { lambda: lam }); }), c.c2, { pointRadius: 3, showLine: false }));
    }
    var o = baseOptions(c, { xType: 'category', legend: ds.length > 1, xTitle: s.xlabel || 'k', ymax: s.ymax });
    f.chart(f.canvas(), { data: { labels: ks, datasets: ds }, options: o });
    f.info.textContent = momentText(d, q);
  };

  function curve(d, q, xs, fn) { return xs.map(function (x) { return { x: x, y: fn(d, x, q) }; }); }

  types.pdf = function (f) {
    var s = f.spec, c = colors(), d = s.dist, q = f.params, r = xRange(s, d, q), xs = linspace(r[0], r[1], 301);
    var main = curve(d, q, xs, Dist.pdf);
    var all = main.map(function (p) { return p.y; });
    var palette = [c.c2, c.c3, c.muted];
    var ds = [lineDs(s.label || '密度', main, c.c1, { fill: 'origin', backgroundColor: c.fill })];
    (s.compare || []).forEach(function (cmp, i) {
      var cd = cmp.dist || d, pts = curve(cd, cmp.params || {}, xs, Dist.pdf);
      all = all.concat(pts.map(function (p) { return p.y; }));
      ds.push(lineDs(cmp.label || cd, pts, palette[i % 3], { borderDash: [6, 4] }));
    });
    var cap = robustMax(all);
    ds.forEach(function (dd) { dd.data = dd.data.map(function (p) { var y = p.y; return { x: p.x, y: (!isFinite(y) || (cap && y > cap)) ? (cap || null) : y }; }); });
    if (ds.length > 1 && !s.label) ds[0].label = '現在の設定';
    var o = baseOptions(c, { legend: ds.length > 1, xmin: r[0], xmax: r[1], ymax: s.ymax || cap, xTitle: s.xlabel || 'x' });
    f.chart(f.canvas(), { data: { datasets: ds }, options: o });
    f.info.textContent = momentText(d, q);
  };

  types.cdf = function (f) {
    var s = f.spec, c = colors(), d = s.dist, q = f.params, r = xRange(s, d, q), disc = Dist.isDiscrete(d);
    var xs = linspace(r[0], r[1], disc ? 601 : 301);
    var ds = [lineDs('F(x)', curve(d, q, xs, Dist.cdf), c.c1, { stepped: false })];
    var info = momentText(d, q);
    if (s.quantile != null) {
      var p = Number(s.quantile), xp = Dist.quantile(d, p, q);
      ds.push(lineDs('p = ' + fmt(p), [{ x: r[0], y: p }, { x: xp, y: p }, { x: xp, y: 0 }], c.c2, { borderDash: [5, 4], pointRadius: [0, 4, 0] }));
      info = 'p = ' + fmt(p) + ' の分位点 x_p = ' + fmt(xp) + '　（' + info + '）';
    }
    var o = baseOptions(c, { legend: ds.length > 1, xmin: r[0], xmax: r[1], ymax: 1.02, xTitle: s.xlabel || 'x' });
    f.chart(f.canvas(), { data: { datasets: ds }, options: o });
    f.info.textContent = info;
  };

  types['pdf-cdf'] = function (f) {
    var s = f.spec, c = colors(), d = s.dist, q = f.params, r = xRange(s, d, q), disc = Dist.isDiscrete(d);
    if (f.x0 == null) f.x0 = Number(((r[0] + r[1]) / 2).toFixed(disc ? 0 : 2));
    var x0 = f.x0, Fx = Dist.cdf(d, x0, q);
    var o1, ds1;
    if (disc) {
      var ks = [], ps = [], cols = [];
      for (var k = Math.ceil(r[0]); k <= Math.floor(r[1]); k++) { ks.push(k); ps.push(Dist.pmf(d, k, q)); cols.push(k <= x0 ? c.c1 : c.fill); }
      ds1 = [{ type: 'bar', label: 'P(X=k)', data: ps, backgroundColor: cols, borderColor: c.c1, borderWidth: 1 }];
      o1 = baseOptions(c, { xType: 'category', xTitle: 'k' });
      f.chart(f.canvas('short'), { data: { labels: ks, datasets: ds1 }, options: o1 });
    } else {
      var xs = linspace(r[0], r[1], 301), pts = curve(d, q, xs, Dist.pdf);
      var cap = robustMax(pts.map(function (p) { return p.y; }));
      pts = pts.map(function (p) { return { x: p.x, y: !isFinite(p.y) ? cap : (cap && p.y > cap ? cap : p.y) }; });
      var left = pts.filter(function (p) { return p.x <= x0; });
      left.push({ x: x0, y: Math.min(cap || Infinity, Dist.pdf(d, x0, q)) });
      ds1 = [lineDs('f(x)', pts, c.c1), lineDs('P(X ≤ x)', left, c.c1, { fill: 'origin', backgroundColor: c.fill, borderWidth: 0 })];
      o1 = baseOptions(c, { xmin: r[0], xmax: r[1], ymax: cap, xTitle: 'x' });
      f.chart(f.canvas('short'), { data: { datasets: ds1 }, options: o1 });
    }
    var xs2 = linspace(r[0], r[1], disc ? 601 : 301);
    var ds2 = [lineDs('F(x)', curve(d, q, xs2, Dist.cdf), c.c1),
      lineDs('F(x₀)', [{ x: r[0], y: Fx }, { x: x0, y: Fx }, { x: x0, y: 0 }], c.c2, { borderDash: [5, 4], pointRadius: [0, 5, 0], pointBackgroundColor: c.c2 })];
    var o2 = baseOptions(c, { xmin: r[0], xmax: r[1], ymax: 1.02, xTitle: 'x' });
    f.chart(f.canvas('short'), { data: { datasets: ds2 }, options: o2 });
    f.info.textContent = 'x = ' + fmt(x0) + ' のとき  P(X ≤ x) = F(x) = ' + fmt(Fx) + '（上の色付き部分の面積' + (disc ? '＝棒の高さの合計' : '') + '）';
  };
  types['pdf-cdf'].controls = function (f, redraw) {
    var s = f.spec, d = s.dist, r = xRange(s, d, f.params), disc = Dist.isDiscrete(d);
    f.x0 = Number(((r[0] + r[1]) / 2).toFixed(disc ? 0 : 2));
    f.slider({ param: 'x', label: 'x の位置', min: r[0], max: r[1], step: disc ? 1 : (r[1] - r[0]) / 200 },
      function () { return f.x0; }, function (v) { f.x0 = v; }, redraw);
  };

  types['survival-hazard'] = function (f) {
    var s = f.spec, c = colors(), d = s.dist, q = f.params, r = xRange(s, d, q);
    var lo = Math.max(r[0], 1e-6), xs = linspace(lo, r[1], 241);
    var S = xs.map(function (x) { return { x: x, y: 1 - Dist.cdf(d, x, q) }; });
    var H = xs.map(function (x) { var sv = 1 - Dist.cdf(d, x, q); return { x: x, y: sv > 1e-9 ? Dist.pdf(d, x, q) / sv : null }; });
    var cap = robustMax(H.map(function (p) { return p.y; }));
    H = H.map(function (p) { return { x: p.x, y: p.y == null ? null : (cap && p.y > cap ? cap : p.y) }; });
    var pair = el('div', 'chart-pair side');
    f.plot.appendChild(pair);
    var mk = function () { var w = el('div', 'chart-canvas short'), cv = document.createElement('canvas'); w.appendChild(cv); pair.appendChild(w); return cv; };
    f.chart(mk(), { data: { datasets: [lineDs('生存関数 S(x)', S, c.c1, { fill: 'origin', backgroundColor: c.fill })] },
      options: baseOptions(c, { legend: true, xmin: r[0], xmax: r[1], ymax: 1.02, xTitle: 'x' }) });
    f.chart(mk(), { data: { datasets: [lineDs('ハザード h(x)', H, c.c2)] },
      options: baseOptions(c, { legend: true, xmin: r[0], xmax: r[1], ymax: cap, xTitle: 'x' }) });
    f.info.textContent = momentText(d, q);
  };

  function corr(xs, ys) {
    var n = xs.length, mx = 0, my = 0, sxx = 0, syy = 0, sxy = 0, i;
    for (i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
    mx /= n; my /= n;
    for (i = 0; i < n; i++) { var a = xs[i] - mx, b = ys[i] - my; sxx += a * a; syy += b * b; sxy += a * b; }
    return { r: sxy / Math.sqrt(sxx * syy), b1: sxy / sxx, b0: my - (sxy / sxx) * mx, mx: mx, my: my };
  }
  function ranks(v) {
    var idx = v.map(function (x, i) { return i; }).sort(function (a, b) { return v[a] - v[b]; }), r = new Array(v.length);
    idx.forEach(function (i, k) { r[i] = k + 1; });
    return r;
  }
  function resid(ys, zs) { var f = corr(zs, ys); return ys.map(function (y, i) { return y - f.b0 - f.b1 * zs[i]; }); }

  types.scatter = function (f) {
    var s = f.spec, c = colors(), q = f.params, n = Math.round(q.n || s.n || 200);
    var rand = mulberry32(f.seed), z = normalRng(rand), xs = [], ys = [], zs = [], i, info;
    var model = s.model;
    if (model === 'bvnorm') {
      var rho = q.rho != null ? q.rho : 0.5;
      for (i = 0; i < n; i++) { var a = z(), b = z(); xs.push(a); ys.push(rho * a + Math.sqrt(1 - rho * rho) * b); }
      info = '母相関 ρ = ' + fmt(rho, 3) + '　標本相関 r = ' + fmt(corr(xs, ys).r, 3) + '　順位相関 rₛ = ' + fmt(corr(ranks(xs), ranks(ys)).r, 3) + '（n = ' + n + '）';
    } else if (model === 'spurious') {
      var ca = q.a != null ? q.a : 1, cb = q.b != null ? q.b : 1;
      for (i = 0; i < n; i++) { var zz = z(); zs.push(zz); xs.push(ca * zz + z()); ys.push(cb * zz + z()); }
      var rr = corr(xs, ys).r, rp = corr(resid(xs, zs), resid(ys, zs)).r;
      info = 'X と Y の標本相関 r = ' + fmt(rr, 3) + '　Z を除いた偏相関 = ' + fmt(rp, 3) + '（理論値 ' + fmt(ca * cb / Math.sqrt((ca * ca + 1) * (cb * cb + 1)), 3) + ' と 0）';
    } else {
      var b0 = q.beta0 != null ? q.beta0 : 1, b1 = q.beta1 != null ? q.beta1 : 0.5, sg = q.sigma != null ? q.sigma : 1;
      for (i = 0; i < n; i++) { var x = rand() * 10; xs.push(x); ys.push(b0 + b1 * x + sg * z()); }
    }
    var pts = xs.map(function (x, i) { return { x: x, y: ys[i] }; });
    var ds = [{ type: 'scatter', label: 'データ', data: pts, backgroundColor: c.c1, pointRadius: n > 300 ? 1.8 : 2.6 }];
    var fit = corr(xs, ys);
    var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
    if (model === 'regression' || s.fit) {
      ds.push(lineDs('最小二乗直線', [{ x: xmin, y: fit.b0 + fit.b1 * xmin }, { x: xmax, y: fit.b0 + fit.b1 * xmax }], c.c2));
      if (model === 'regression') info = '推定値 b0 = ' + fmt(fit.b0, 3) + ', b1 = ' + fmt(fit.b1, 3) + '　決定係数 R² = ' + fmt(fit.r * fit.r, 3) + '（n = ' + n + '）';
    }
    var o = baseOptions(c, { legend: ds.length > 1, ymin: null, xTitle: s.xlabel || 'x', yTitle: s.ylabel || 'y' });
    o.interaction = { mode: 'nearest', intersect: true };
    var cv = f.canvas('tall');
    f.chart(cv, { data: { datasets: ds }, options: o });
    f.info.textContent = info;
  };
  types.scatter.controls = function (f, redraw) {
    f.button('もう一度（乱数を変える）', function () { f.seed = (f.seed * 1103515245 + 12345) >>> 0; redraw(); });
  };

  types.custom = function (f) {
    var fn = customs[f.spec.id];
    if (!fn) { f.missing('この図（' + f.spec.id + '）は準備中です'); return; }
    fn(api(f), f.params, colors());
  };

  // カスタム図に渡す道具
  function api(f) {
    return {
      el: f.plot, info: f.info, seed: f.seed, rand: mulberry32(f.seed), colors: colors, fmt: fmt,
      chartjs: hasChartJs(), baseOptions: baseOptions, lineDs: lineDs, linspace: linspace, Dist: Dist,
      normal: normalRng(mulberry32(f.seed + 7)),
      hist: hist,
      chart: function (cls, config) {
        if (!hasChartJs()) throw new Error('グラフ用ライブラリを読み込み中です（オフラインでは表示できません）');
        return f.chart(f.canvas(cls), config);
      },
      canvas: function (height) {
        var wrap = el('div', 'chart-canvas');
        wrap.style.height = (height || 260) + 'px';
        var cv = document.createElement('canvas');
        wrap.appendChild(cv);
        f.plot.appendChild(wrap);
        var w = wrap.clientWidth || 600, h = height || 260, dpr = root.devicePixelRatio || 1;
        cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
        cv.style.width = w + 'px'; cv.style.height = h + 'px';
        var ctx = cv.getContext('2d');
        ctx.scale(dpr, dpr);
        ctx.font = '12px ' + colors().font;
        return { canvas: cv, ctx: ctx, w: w, h: h };
      },
      redraw: function () { f.redraw(); },
      _track: function (ch) { f.charts.push(ch); }
    };
  }

  // ---------- 公開 API ----------
  function render(host, spec) {
    var f = new Frame(host, spec);
    var typ = types[spec.type];
    var needsChartJs = spec.type !== 'custom';
    f.redraw = function () {
      f.clear();
      if (!typ) { f.missing('未対応の図の種類です: ' + spec.type); return; }
      if (needsChartJs && !hasChartJs()) { f.missing('グラフ用ライブラリを読み込み中です（オフラインでは表示できません）'); return; }
      try { typ(f); } catch (e) { f.missing('図を描けませんでした: ' + e.message); if (root.console) console.error(e); }
    };
    var pending = false;
    var schedule = function () {
      if (pending) return;
      pending = true;
      (root.requestAnimationFrame || setTimeout)(function () { pending = false; f.redraw(); });
    };
    if (typ && typ.controls) typ.controls(f, schedule);
    if (spec.type === 'custom' && randomIds[spec.id]) types.scatter.controls(f, schedule);
    f.paramSliders(schedule);
    f.redraw();
    if (spec.type === 'custom' && root.ResizeObserver) {
      var lastW = 0;
      f.ro = new ResizeObserver(function (ents) { var w = ents[0].contentRect.width; if (Math.abs(w - lastW) > 4) { lastW = w; schedule(); } });
      f.ro.observe(f.fig);
    }
    live.push(f);
    return f;
  }
  function destroyAll() {
    live.forEach(function (f) { f.clear(); if (f.ro) f.ro.disconnect(); });
    live = [];
  }
  function redrawAll() { live.forEach(function (f) { f.redraw(); }); }
  function registerCustom(id, fn, random) { customs[id] = fn; if (random) randomIds[id] = true; }

  root.Charts = { _rng: mulberry32, _normal: normalRng, render: render, destroyAll: destroyAll, redrawAll: redrawAll, registerCustom: registerCustom, has: function (id) { return !!customs[id]; } };

  // ---------- カスタム図 ----------

  // ベイズの面積図：横＝事前確率 P(A)、縦＝条件付き確率 P(B|A), P(B|Aᶜ)
  // params: prior, sensitivity (=P(B|A)), specificity (=P(Bᶜ|Aᶜ)), 省略可のラベル labelA, labelNotA, labelB
  registerCustom('bayes-area', function (a, p, c) {
    var prior = p.prior != null ? p.prior : 0.01, se = p.sensitivity != null ? p.sensitivity : 0.99, sp = p.specificity != null ? p.specificity : 0.95;
    var LA = p.labelA || '病気', LN = p.labelNotA || '健康', LB = p.labelB || '陽性';
    var cv = a.canvas(280), ctx = cv.ctx, W = cv.w, H = cv.h, pad = 28, w = W - pad * 2, h = H - pad - 22;
    var x0 = pad, y0 = 22, wd = Math.min(w - 3, Math.max(3, w * prior)), wh = w - wd;
    ctx.fillStyle = c.bg; ctx.fillRect(0, 0, W, H);
    // 左の列（A）：上が B
    ctx.fillStyle = c.c1; ctx.fillRect(x0, y0, wd, h * se);
    ctx.fillStyle = c.line; ctx.fillRect(x0, y0 + h * se, wd, h * (1 - se));
    // 右の列（Aᶜ）：上が B
    ctx.fillStyle = c.c2; ctx.fillRect(x0 + wd, y0, wh, h * (1 - sp));
    ctx.fillStyle = c.surface; ctx.fillRect(x0 + wd, y0 + h * (1 - sp), wh, h * sp);
    ctx.strokeStyle = c.muted; ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, w, h);
    ctx.beginPath(); ctx.moveTo(x0 + wd, y0); ctx.lineTo(x0 + wd, y0 + h); ctx.stroke();
    // 境目の高さを左右で見比べやすいよう点線を引く
    ctx.setLineDash([4, 4]); ctx.strokeStyle = c.fg;
    ctx.beginPath(); ctx.moveTo(x0, y0 + h * se); ctx.lineTo(x0 + w, y0 + h * se); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = c.fg; ctx.textBaseline = 'bottom';
    ctx.fillText(LA + ' ' + a.fmt(prior * 100, 2) + '%', x0, y0 - 4);
    ctx.textAlign = 'right'; ctx.fillText(LN + ' ' + a.fmt((1 - prior) * 100, 2) + '%', x0 + w, y0 - 4);
    ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = c.muted;
    ctx.fillText('色付き＝' + LB + '（濃い色：' + LA + 'かつ' + LB + '　薄い色：' + LN + 'かつ' + LB + '）　点線＝左の列の境目', x0, y0 + h + 6);
    var tp = prior * se, fp = (1 - prior) * (1 - sp), pos = tp + fp;
    a.info.textContent = 'P(' + LB + '|' + LA + ') = ' + a.fmt(se) + '，P(' + LB + '|' + LN + ') = ' + a.fmt(1 - sp) +
      '　P(' + LB + ') = ' + a.fmt(tp) + ' ＋ ' + a.fmt(fp) + ' = ' + a.fmt(pos) + '　→　P(' + LA + '|' + LB + ') = ' + a.fmt(pos > 0 ? tp / pos : NaN);
  });

  // 同時分布表と周辺分布。行をタップすると条件付き分布を表示
  registerCustom('joint-marginal', function (a, p) {
    var T = p.table || [[0.1, 0.2, 0.1], [0.2, 0.3, 0.1]];
    var rl = p.rowLabels || T.map(function (_, i) { return String(i); });
    var cl = p.colLabels || T[0].map(function (_, j) { return String(j); });
    var rn = p.rowName || 'X', cn = p.colName || 'Y';
    var rs = T.map(function (r) { return r.reduce(function (s, v) { return s + v; }, 0); });
    var cs = cl.map(function (_, j) { return T.reduce(function (s, r) { return s + r[j]; }, 0); });
    var tbl = document.createElement('table');
    tbl.className = 'jm-table';
    var thead = '<thead><tr><th>' + rn + ' ＼ ' + cn + '</th>' + cl.map(function (l) { return '<th>' + l + '</th>'; }).join('') + '<th>' + rn + ' の周辺</th></tr></thead>';
    var body = '<tbody>' + T.map(function (r, i) {
      return '<tr data-i="' + i + '"><th>' + rl[i] + '</th>' + r.map(function (v) { return '<td>' + a.fmt(v) + '</td>'; }).join('') + '<td class="marg">' + a.fmt(rs[i]) + '</td></tr>';
    }).join('') + '</tbody>';
    var foot = '<tfoot><tr><th>' + cn + ' の周辺</th>' + cs.map(function (v) { return '<td class="marg">' + a.fmt(v) + '</td>'; }).join('') + '<td class="marg">' + a.fmt(rs.reduce(function (s, v) { return s + v; }, 0)) + '</td></tr></tfoot>';
    tbl.innerHTML = thead + body + foot;
    var wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    wrap.style.border = '0';
    wrap.appendChild(tbl);
    a.el.appendChild(wrap);
    function show(i) {
      Array.prototype.forEach.call(tbl.querySelectorAll('tbody tr'), function (tr) { tr.classList.toggle('sel', Number(tr.dataset.i) === i); });
      a.info.textContent = 'P(' + cn + ' | ' + rn + '=' + rl[i] + ') = ' + T[i].map(function (v, j) { return cl[j] + ': ' + a.fmt(v / rs[i]); }).join('，') + '（行の値 ÷ 行の合計 ' + a.fmt(rs[i]) + '）';
    }
    tbl.addEventListener('click', function (e) { var tr = e.target.closest('tbody tr'); if (tr) show(Number(tr.dataset.i)); });
    show(0);
  });

  // 2つの正の数の相加平均・幾何平均・調和平均を数直線に並べる
  registerCustom('mean-inequality', function (a, p, c) {
    var x = Math.max(1e-9, p.a != null ? p.a : 2), y = Math.max(1e-9, p.b != null ? p.b : 8);
    var am = (x + y) / 2, gm = Math.sqrt(x * y), hm = 2 * x * y / (x + y);
    var cv = a.canvas(170), ctx = cv.ctx, W = cv.w, pad = 30, lo = 0, hi = Math.max(x, y) * 1.08;
    var X = function (v) { return pad + (W - 2 * pad) * (v - lo) / (hi - lo); }, base = 100;
    ctx.strokeStyle = c.muted; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pad, base); ctx.lineTo(W - pad, base); ctx.stroke();
    ctx.fillStyle = c.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var k = 0; k <= 4; k++) { var v = lo + (hi - lo) * k / 4; ctx.fillRect(X(v) - 0.5, base, 1, 5); ctx.fillText(a.fmt(v, 2), X(v), base + 8); }
    function dot(v, col, label, row) {
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(v), base, 6, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(X(v), base - 8); ctx.lineTo(X(v), base - 18 - row * 22); ctx.stroke();
      ctx.textBaseline = 'bottom'; ctx.fillText(label, X(v), base - 20 - row * 22);
    }
    ctx.textAlign = 'center';
    dot(x, c.muted, 'a', 0); dot(y, c.muted, 'b', 0);
    dot(hm, c.c3, 'HM', 1); dot(gm, c.c2, 'GM', 2); dot(am, c.c1, 'AM', 3);
    a.info.textContent = 'AM = ' + a.fmt(am) + ' ≥ GM = ' + a.fmt(gm) + ' ≥ HM = ' + a.fmt(hm) + '　（GM² = AM × HM = ' + a.fmt(am * hm) + '）';
  });
})(typeof globalThis !== 'undefined' ? globalThis : this);
