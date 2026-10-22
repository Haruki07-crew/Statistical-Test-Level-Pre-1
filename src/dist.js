/* 分布の密度・確率・累積分布の計算ライブラリ（外部依存なし） */
(function (root) {
  'use strict';

  // ---------- 特殊関数 ----------
  var LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

  function lgamma(x) {
    if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
    x -= 1;
    var a = LANCZOS[0], t = x + 7.5;
    for (var i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  function lbeta(a, b) { return lgamma(a) + lgamma(b) - lgamma(a + b); }
  function lchoose(n, k) { return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1); }

  // 正則化された下側不完全ガンマ関数 P(a, x)
  function gammainc(a, x) {
    if (x <= 0) return 0;
    if (!isFinite(x)) return 1;
    var lpre = a * Math.log(x) - x - lgamma(a);
    if (x < a + 1) {
      var sum = 1 / a, term = sum, ap = a;
      for (var n = 0; n < 10000; n++) {
        ap += 1; term *= x / ap; sum += term;
        if (Math.abs(term) < Math.abs(sum) * 1e-16) break;
      }
      return Math.min(1, sum * Math.exp(lpre));
    }
    return 1 - gammaincUpperCF(a, x, lpre);
  }
  function gammaincUpperCF(a, x, lpre) {
    var tiny = 1e-300, b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
    for (var i = 1; i < 10000; i++) {
      var an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < tiny) d = tiny;
      c = b + an / c; if (Math.abs(c) < tiny) c = tiny;
      d = 1 / d;
      var del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-16) break;
    }
    return Math.exp(lpre) * h;
  }
  // 上側 Q(a,x) を精度よく
  function gammaincc(a, x) {
    if (x <= 0) return 1;
    if (x < a + 1) return 1 - gammainc(a, x);
    return gammaincUpperCF(a, x, a * Math.log(x) - x - lgamma(a));
  }

  // 正則化された不完全ベータ関数 I_x(a, b)
  function betacf(a, b, x) {
    var tiny = 1e-300, qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < tiny) d = tiny;
    d = 1 / d;
    var h = d;
    for (var m = 1; m < 10000; m++) {
      var m2 = 2 * m, aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
      c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
      c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
      d = 1 / d;
      var del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-16) break;
    }
    return h;
  }
  function betainc(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    var lbt = a * Math.log(x) + b * Math.log(1 - x) - lbeta(a, b);
    if (x < (a + 1) / (a + b + 2)) return Math.exp(lbt) * betacf(a, b, x) / a;
    return 1 - Math.exp(lbt) * betacf(b, a, 1 - x) / b;
  }

  function normCdfStd(z) {
    // Φ(z) = P(1/2, z^2/2) を使って両側とも精度を保つ
    var q = 0.5 * gammaincc(0.5, z * z / 2);
    return z < 0 ? q : 1 - q;
  }
  function normPdfStd(z) { return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); }
  function erf(x) { return x >= 0 ? gammainc(0.5, x * x) : -gammainc(0.5, x * x); }

  function poisLogW(j, lam) { return lam > 0 ? -lam + j * Math.log(lam) - lgamma(j + 1) : (j === 0 ? 0 : -Infinity); }
  // ポアソン重み付き和 Σ w_j f(j)（重みが十分小さくなるまで、最頻値から両側へ）
  function poisMix(lam, f) {
    if (lam <= 0) return f(0);
    var mode = Math.floor(lam), s = 0, j, w;
    for (j = mode; j < mode + 2000; j++) {
      w = Math.exp(poisLogW(j, lam)); s += w * f(j);
      if (j > lam && w < 1e-17) break;
    }
    for (j = mode - 1; j >= 0; j--) {
      w = Math.exp(poisLogW(j, lam)); s += w * f(j);
      if (w < 1e-17) break;
    }
    return s;
  }

  // ---------- 分布の定義 ----------
  function chi2pdf(x, k) {
    if (x < 0) return 0;
    if (x === 0) return k < 2 ? Infinity : (k === 2 ? 0.5 : 0);
    return Math.exp((k / 2 - 1) * Math.log(x) - x / 2 - (k / 2) * Math.LN2 - lgamma(k / 2));
  }
  function betapdf(x, a, b) {
    if (x < 0 || x > 1) return 0;
    if (x === 0) return a < 1 ? Infinity : (a === 1 ? Math.exp(-lbeta(a, b)) : 0);
    if (x === 1) return b < 1 ? Infinity : (b === 1 ? Math.exp(-lbeta(a, b)) : 0);
    return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta(a, b));
  }
  function tpdf(x, v) { return Math.exp(lgamma((v + 1) / 2) - lgamma(v / 2) - 0.5 * Math.log(v * Math.PI) - (v + 1) / 2 * Math.log(1 + x * x / v)); }
  function tcdf(x, v) {
    var ib = betainc(v / 2, 0.5, v / (v + x * x));
    return x >= 0 ? 1 - 0.5 * ib : 0.5 * ib;
  }
  function fpdf(x, d1, d2) {
    if (x < 0) return 0;
    if (x === 0) return d1 < 2 ? Infinity : (d1 === 2 ? 1 : 0);
    return Math.exp(0.5 * (d1 * Math.log(d1 * x) + d2 * Math.log(d2) - (d1 + d2) * Math.log(d1 * x + d2)) - Math.log(x) - lbeta(d1 / 2, d2 / 2));
  }
  function fcdf(x, d1, d2) { return x <= 0 ? 0 : betainc(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2)); }

  // 非心t：T = (Z + δ)/sqrt(V/ν)。W = sqrt(V/ν) の密度で数値積分（シンプソン則）
  function nctIntegral(x, v, nc, wantCdf) {
    var lc = (v / 2) * Math.log(v) - (v / 2 - 1) * Math.LN2 - lgamma(v / 2);
    var wmax = 1 + 12 / Math.sqrt(v) + 2, n = 4000, h = wmax / n, s = 0;
    for (var i = 0; i <= n; i++) {
      var w = i * h, g;
      if (w === 0) g = v === 1 ? Math.exp(lc) : 0;
      else g = Math.exp(lc + (v - 1) * Math.log(w) - v * w * w / 2);
      var val = wantCdf ? g * normCdfStd(x * w - nc) : g * w * normPdfStd(x * w - nc);
      s += val * (i === 0 || i === n ? 1 : (i % 2 ? 4 : 2));
    }
    return s * h / 3;
  }

  var D = {
    bernoulli: { discrete: true, params: ['p'], defaults: { p: 0.3 },
      pmf: function (k, q) { return k === 0 ? 1 - q.p : (k === 1 ? q.p : 0); },
      support: function () { return [0, 1]; } },
    binom: { discrete: true, params: ['n', 'p'], defaults: { n: 10, p: 0.5 },
      pmf: function (k, q) {
        if (k < 0 || k > q.n) return 0;
        if (q.p === 0) return k === 0 ? 1 : 0;
        if (q.p === 1) return k === q.n ? 1 : 0;
        return Math.exp(lchoose(q.n, k) + k * Math.log(q.p) + (q.n - k) * Math.log(1 - q.p));
      },
      support: function (q) { return [0, q.n]; } },
    hypergeom: { discrete: true, params: ['N', 'K', 'n'], defaults: { N: 50, K: 10, n: 10 },
      pmf: function (k, q) {
        if (k < Math.max(0, q.n - (q.N - q.K)) || k > Math.min(q.n, q.K)) return 0;
        return Math.exp(lchoose(q.K, k) + lchoose(q.N - q.K, q.n - k) - lchoose(q.N, q.n));
      },
      support: function (q) { return [Math.max(0, q.n - (q.N - q.K)), Math.min(q.n, q.K)]; } },
    poisson: { discrete: true, params: ['lambda'], defaults: { lambda: 3 },
      pmf: function (k, q) { return k < 0 ? 0 : Math.exp(poisLogW(k, q.lambda)); },
      support: function (q) { return [0, Math.ceil(q.lambda + 6 * Math.sqrt(q.lambda) + 6)]; } },
    geom: { discrete: true, params: ['p', 'start'], defaults: { p: 0.3, start: 0 },
      pmf: function (k, q) { var s = q.start || 0, j = k - s; return j < 0 ? 0 : q.p * Math.pow(1 - q.p, j); },
      support: function (q) { var s = q.start || 0; return [s, s + Math.ceil(Math.log(1e-4) / Math.log(1 - q.p))]; } },
    nbinom: { discrete: true, params: ['r', 'p'], defaults: { r: 3, p: 0.4 },
      pmf: function (k, q) { return k < 0 ? 0 : Math.exp(lgamma(k + q.r) - lgamma(k + 1) - lgamma(q.r) + q.r * Math.log(q.p) + k * Math.log(1 - q.p)); },
      support: function (q) { var m = q.r * (1 - q.p) / q.p, s = Math.sqrt(m / q.p); return [0, Math.ceil(m + 6 * s + 5)]; } },
    unif: { params: ['a', 'b'], defaults: { a: 0, b: 1 },
      pdf: function (x, q) { return x < q.a || x > q.b ? 0 : 1 / (q.b - q.a); },
      cdf: function (x, q) { return x <= q.a ? 0 : (x >= q.b ? 1 : (x - q.a) / (q.b - q.a)); },
      range: function (q) { var w = q.b - q.a; return [q.a - 0.2 * w, q.b + 0.2 * w]; } },
    norm: { params: ['mu', 'sigma'], defaults: { mu: 0, sigma: 1 },
      pdf: function (x, q) { return normPdfStd((x - q.mu) / q.sigma) / q.sigma; },
      cdf: function (x, q) { return normCdfStd((x - q.mu) / q.sigma); },
      range: function (q) { return [q.mu - 4 * q.sigma, q.mu + 4 * q.sigma]; } },
    lognorm: { params: ['mu', 'sigma'], defaults: { mu: 0, sigma: 0.5 },
      pdf: function (x, q) { return x <= 0 ? 0 : normPdfStd((Math.log(x) - q.mu) / q.sigma) / (q.sigma * x); },
      cdf: function (x, q) { return x <= 0 ? 0 : normCdfStd((Math.log(x) - q.mu) / q.sigma); },
      range: function (q) { return [0, Math.exp(q.mu + 3 * q.sigma)]; } },
    expon: { params: ['rate'], defaults: { rate: 1 },
      pdf: function (x, q) { return x < 0 ? 0 : q.rate * Math.exp(-q.rate * x); },
      cdf: function (x, q) { return x <= 0 ? 0 : -Math.expm1(-q.rate * x); },
      range: function (q) { return [0, 6 / q.rate]; } },
    gamma: { params: ['shape', 'scale'], defaults: { shape: 2, scale: 1 },
      pdf: function (x, q) {
        if (x < 0) return 0;
        if (x === 0) return q.shape < 1 ? Infinity : (q.shape === 1 ? 1 / q.scale : 0);
        return Math.exp((q.shape - 1) * Math.log(x) - x / q.scale - lgamma(q.shape) - q.shape * Math.log(q.scale));
      },
      cdf: function (x, q) { return gammainc(q.shape, x / q.scale); },
      range: function (q) { return [0, q.scale * (q.shape + 5 * Math.sqrt(q.shape) + 2)]; } },
    beta: { params: ['a', 'b'], defaults: { a: 2, b: 2 },
      pdf: function (x, q) { return betapdf(x, q.a, q.b); },
      cdf: function (x, q) { return betainc(q.a, q.b, x); },
      range: function () { return [0, 1]; } },
    cauchy: { params: ['loc', 'scale'], defaults: { loc: 0, scale: 1 },
      pdf: function (x, q) { var z = (x - q.loc) / q.scale; return 1 / (Math.PI * q.scale * (1 + z * z)); },
      cdf: function (x, q) { return 0.5 + Math.atan((x - q.loc) / q.scale) / Math.PI; },
      range: function (q) { return [q.loc - 8 * q.scale, q.loc + 8 * q.scale]; } },
    weibull: { params: ['shape', 'scale'], defaults: { shape: 1.5, scale: 1 },
      pdf: function (x, q) {
        if (x < 0) return 0;
        var z = x / q.scale;
        if (x === 0) return q.shape < 1 ? Infinity : (q.shape === 1 ? 1 / q.scale : 0);
        return (q.shape / q.scale) * Math.pow(z, q.shape - 1) * Math.exp(-Math.pow(z, q.shape));
      },
      cdf: function (x, q) { return x <= 0 ? 0 : -Math.expm1(-Math.pow(x / q.scale, q.shape)); },
      range: function (q) { return [0, q.scale * Math.pow(Math.log(1e4), 1 / q.shape)]; } },
    logistic: { params: ['loc', 'scale'], defaults: { loc: 0, scale: 1 },
      pdf: function (x, q) { var e = Math.exp(-Math.abs(x - q.loc) / q.scale); return e / (q.scale * (1 + e) * (1 + e)); },
      cdf: function (x, q) { return 1 / (1 + Math.exp(-(x - q.loc) / q.scale)); },
      range: function (q) { return [q.loc - 8 * q.scale, q.loc + 8 * q.scale]; } },
    chi2: { params: ['df'], defaults: { df: 3 },
      pdf: function (x, q) { return chi2pdf(x, q.df); },
      cdf: function (x, q) { return gammainc(q.df / 2, x / 2); },
      range: function (q) { return [0, q.df + 5 * Math.sqrt(2 * q.df) + 3]; } },
    t: { params: ['df'], defaults: { df: 5 },
      pdf: function (x, q) { return tpdf(x, q.df); },
      cdf: function (x, q) { return tcdf(x, q.df); },
      range: function () { return [-5, 5]; } },
    f: { params: ['df1', 'df2'], defaults: { df1: 5, df2: 10 },
      pdf: function (x, q) { return fpdf(x, q.df1, q.df2); },
      cdf: function (x, q) { return fcdf(x, q.df1, q.df2); },
      range: function () { return [0, 5]; } },
    ncx2: { params: ['df', 'nc'], defaults: { df: 3, nc: 2 },
      pdf: function (x, q) { return x < 0 ? 0 : poisMix(q.nc / 2, function (j) { return chi2pdf(x, q.df + 2 * j); }); },
      cdf: function (x, q) { return x <= 0 ? 0 : poisMix(q.nc / 2, function (j) { return gammainc(q.df / 2 + j, x / 2); }); },
      range: function (q) { var m = q.df + q.nc; return [0, m + 5 * Math.sqrt(2 * (q.df + 2 * q.nc)) + 3]; } },
    nct: { params: ['df', 'nc'], defaults: { df: 10, nc: 1 },
      pdf: function (x, q) { return nctIntegral(x, q.df, q.nc, false); },
      cdf: function (x, q) { return Math.min(1, Math.max(0, nctIntegral(x, q.df, q.nc, true))); },
      range: function (q) { return [Math.min(-4, q.nc - 4), Math.max(4, q.nc + 5)]; } },
    ncf: { params: ['df1', 'df2', 'nc'], defaults: { df1: 3, df2: 20, nc: 3 },
      pdf: function (x, q) {
        if (x <= 0) return 0;
        var d1 = q.df1, d2 = q.df2, u = d1 * x / (d2 + d1 * x), du = d1 * d2 / Math.pow(d2 + d1 * x, 2);
        return poisMix(q.nc / 2, function (j) { return betapdf(u, d1 / 2 + j, d2 / 2); }) * du;
      },
      cdf: function (x, q) {
        if (x <= 0) return 0;
        var u = q.df1 * x / (q.df2 + q.df1 * x);
        return poisMix(q.nc / 2, function (j) { return betainc(q.df1 / 2 + j, q.df2 / 2, u); });
      },
      range: function (q) { return [0, Math.max(5, 3 * (1 + q.nc / q.df1))]; } },
    normmix: { params: ['w', 'mu1', 'sigma1', 'mu2', 'sigma2'], defaults: { w: 0.5, mu1: -1.5, sigma1: 1, mu2: 1.5, sigma2: 1 },
      pdf: function (x, q) { return q.w * normPdfStd((x - q.mu1) / q.sigma1) / q.sigma1 + (1 - q.w) * normPdfStd((x - q.mu2) / q.sigma2) / q.sigma2; },
      cdf: function (x, q) { return q.w * normCdfStd((x - q.mu1) / q.sigma1) + (1 - q.w) * normCdfStd((x - q.mu2) / q.sigma2); },
      range: function (q) { return [Math.min(q.mu1 - 4 * q.sigma1, q.mu2 - 4 * q.sigma2), Math.max(q.mu1 + 4 * q.sigma1, q.mu2 + 4 * q.sigma2)]; } }
  };

  function get(name) {
    var d = D[name];
    if (!d) throw new Error('未知の分布: ' + name);
    return d;
  }
  function withDefaults(name, params) {
    var d = get(name), q = {};
    for (var k in d.defaults) q[k] = d.defaults[k];
    for (var k2 in params || {}) q[k2] = Number(params[k2]);
    return q;
  }

  var Dist = {
    names: Object.keys(D),
    isDiscrete: function (name) { return !!get(name).discrete; },
    params: function (name) { return get(name).params.slice(); },
    withDefaults: withDefaults,
    // 確率関数（離散）または密度（連続）
    pdf: function (name, x, params) {
      var d = get(name), q = withDefaults(name, params);
      if (d.discrete) return (Math.round(x) === x) ? d.pmf(x, q) : 0;
      return d.pdf(x, q);
    },
    pmf: function (name, k, params) { return Dist.pdf(name, k, params); },
    cdf: function (name, x, params) {
      var d = get(name), q = withDefaults(name, params);
      if (!d.discrete) return d.cdf(x, q);
      var sup = d.support(q), s = 0, top = Math.floor(x);
      if (top < sup[0]) return 0;
      for (var k = sup[0]; k <= top; k++) {
        var p = d.pmf(k, q); s += p;
        if (k > sup[1] && p < 1e-16) break;
      }
      return Math.min(1, s);
    },
    // 描画用の範囲（連続）または台（離散）
    range: function (name, params) {
      var d = get(name), q = withDefaults(name, params);
      return d.discrete ? d.support(q) : d.range(q);
    },
    quantile: function (name, p, params) {
      var d = get(name), q = withDefaults(name, params);
      if (d.discrete) {
        var sup = d.support(q), s = 0;
        for (var k = sup[0]; k < sup[0] + 100000; k++) { s += d.pmf(k, q); if (s >= p - 1e-12) return k; }
        return NaN;
      }
      var r = d.range(q), lo = r[0], hi = r[1], i;
      for (i = 0; i < 200 && d.cdf(lo, q) > p; i++) lo -= (hi - lo);
      for (i = 0; i < 200 && d.cdf(hi, q) < p; i++) hi += (hi - lo);
      for (i = 0; i < 200; i++) {
        var mid = (lo + hi) / 2;
        if (d.cdf(mid, q) < p) lo = mid; else hi = mid;
        if (hi - lo < 1e-12 * (1 + Math.abs(mid))) break;
      }
      return (lo + hi) / 2;
    },
    // 平均と分散（数値計算）
    moments: function (name, params) {
      var d = get(name), q = withDefaults(name, params), m = 0, m2 = 0, k, p;
      if (d.discrete) {
        var sup = d.support(q);
        for (k = sup[0]; k <= sup[1] + 2000; k++) { p = d.pmf(k, q); m += k * p; m2 += k * k * p; if (k > sup[1] && p < 1e-17) break; }
      } else {
        var r = d.range(q), n = 4000, h = (r[1] - r[0]) / n;
        for (var i = 0; i <= n; i++) { var x = r[0] + i * h; p = d.pdf(x, q); if (!isFinite(p)) continue; var w = (i === 0 || i === n) ? 0.5 : 1; m += w * x * p * h; m2 += w * x * x * p * h; }
      }
      return { mean: m, variance: m2 - m * m };
    },
    special: { lgamma: lgamma, gammainc: gammainc, gammaincc: gammaincc, betainc: betainc, erf: erf, normCdf: normCdfStd, normPdf: normPdfStd }
  };

  root.Dist = Dist;
})(typeof globalThis !== 'undefined' ? globalThis : this);
