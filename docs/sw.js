/* オフラインでも開けるようにする service worker（build.py が ac58283f7694 を index.html のハッシュに差し替える）
   ・キャッシュ名に版を付け、新しい版が入ったら古いキャッシュを消してすぐ切り替える
   ・ページはネット優先（HTTP キャッシュも確かめ直す）。つながらなければ保存した版を出す
   ・MathJax（本体と最初に読み込む拡張）と Chart.js も保存し、オフラインでも数式とグラフを描けるようにする
   ・準1級（./）・2級（./g2/）・教科書（./tb/）のページを、それぞれのパスで保存する */
var CACHE = 'jun1-dict-ac58283f7694';
var CORE = ['./', './index.html', './manifest.json', './icon.svg', './icon-192.png', './g2/', './g2/index.html', './g2/manifest.json', './tb/', './tb/index.html', './tb/manifest.json'];
var CDN = [
  'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js',
  'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/input/tex/extensions/boldsymbol.js',
  'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js'
];
self.addEventListener('install', function (e) {
  e.waitUntil(caches.open(CACHE).then(function (c) {
    return Promise.all(CORE.map(function (u) { return c.add(new Request(u, { cache: 'reload' })); })).then(function () {
      return Promise.all(CDN.map(function (u) { return c.add(new Request(u, { mode: 'cors' })).catch(function () {}); }));
    });
  }).then(function () { return self.skipWaiting(); }));
});
self.addEventListener('activate', function (e) {
  e.waitUntil(caches.keys().then(function (ks) {
    return Promise.all(ks.filter(function (k) { return k.indexOf('jun1-dict-') === 0 && k !== CACHE; }).map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});
self.addEventListener('fetch', function (e) {
  var req = e.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  var same = url.origin === self.location.origin;
  if (same && (req.mode === 'navigate' || /\/(index\.html)?$/.test(url.pathname))) {
    var lv = /\/(g2|tb)\/(index\.html)?$/.exec(url.pathname), page = lv ? './' + lv[1] + '/index.html' : './index.html';
    e.respondWith(fetch(req.url, { cache: 'no-cache', credentials: 'same-origin' }).then(function (res) {
      if (res && res.ok) { var copy = res.clone(); caches.open(CACHE).then(function (c) { c.put(page, copy); }); }
      return res;
    }).catch(function () { return caches.match(page); }));
    return;
  }
  if (same || /cdn\.jsdelivr\.net|fonts\.(googleapis|gstatic)\.com/.test(url.host)) {
    e.respondWith(caches.match(req, { ignoreSearch: same }).then(function (hit) {
      var net = fetch(req).then(function (res) {
        if (res && (res.ok || res.type === 'opaque')) { var copy = res.clone(); caches.open(CACHE).then(function (c) { c.put(req, copy); }); }
        return res;
      }).catch(function () { return hit; });
      return hit || net;
    }));
  }
});
