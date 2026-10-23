/* オフラインでも開けるようにする service worker（build.py が c40efc4ae8ec を差し替える） */
var CACHE = 'jun1-dict-c40efc4ae8ec';
var CORE = ['./', './index.html', './manifest.json', './icon.svg', './icon-192.png'];
var CDN = [
  'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js',
  'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js'
];
self.addEventListener('install', function (e) {
  e.waitUntil(caches.open(CACHE).then(function (c) {
    return c.addAll(CORE).then(function () {
      return Promise.all(CDN.map(function (u) { return c.add(new Request(u, { mode: 'cors' })).catch(function () {}); }));
    });
  }).then(function () { return self.skipWaiting(); }));
});
self.addEventListener('activate', function (e) {
  e.waitUntil(caches.keys().then(function (ks) {
    return Promise.all(ks.filter(function (k) { return k.indexOf('jun1-dict-') === 0 && k !== CACHE; }).map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});
// ページはネットを優先し（新しい版をすぐ反映）、つながらなければ保存した版を出す。CDN とフォントは保存した版を優先する
self.addEventListener('fetch', function (e) {
  var req = e.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  var same = url.origin === self.location.origin;
  if (same && (req.mode === 'navigate' || /\/(index\.html)?$/.test(url.pathname))) {
    e.respondWith(fetch(req).then(function (res) {
      var copy = res.clone();
      caches.open(CACHE).then(function (c) { c.put('./index.html', copy); });
      return res;
    }).catch(function () { return caches.match('./index.html'); }));
    return;
  }
  if (same || /cdn\.jsdelivr\.net|fonts\.(googleapis|gstatic)\.com/.test(url.host)) {
    e.respondWith(caches.match(req).then(function (hit) {
      var net = fetch(req).then(function (res) {
        if (res && (res.ok || res.type === 'opaque')) { var copy = res.clone(); caches.open(CACHE).then(function (c) { c.put(req, copy); }); }
        return res;
      }).catch(function () { return hit; });
      return hit || net;
    }));
  }
});
