/* Free AI Video Studio — minimal service worker (offline splash only) */
var CACHE = 'fav-studio-v1';
var ASSETS = ['./index.html', './manifest.json', './icon-192.png', './icon-512.png'];

self.addEventListener('install', function (e) {
  e.waitUntil(caches.open(CACHE).then(function (c) { return c.addAll(ASSETS); }));
  self.skipWaiting();
});
self.addEventListener('activate', function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) { return k !== CACHE; }).map(function (k) {
        return caches.delete(k);
      }));
    })
  );
  self.clients.claim();
});
self.addEventListener('fetch', function (e) {
  var url = new URL(e.request.url);
  // Only cache our own shell files; the Streamlit app stays network-only.
  if (url.pathname.endsWith('/pwa/') || ASSETS.some(function (a) { return url.pathname.endsWith(a.slice(1)); })) {
    e.respondWith(
      caches.match(e.request).then(function (hit) { return hit || fetch(e.request); })
    );
  }
});
