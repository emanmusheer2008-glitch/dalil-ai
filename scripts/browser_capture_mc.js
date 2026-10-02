/*
 * Dalil AI — reproducible capture of the Ministry of Commerce e-services catalogue.
 *
 * HOW IT WAS USED
 *   1. Open https://mc.gov.sa/en/eservices/Pages/default.aspx in a normal browser.
 *   2. Paste this script in the browser DevTools console.
 *   3. Wait for "done" (≈150 pages at one request every ~2 s; the server set the pace).
 *   4. Run  __dalilExport()  to save mc_services_capture.json, then move it to
 *      data/raw/official/mc/ and run  python -m src.indexing.build_index
 *
 * WHY A BROWSER SCRIPT
 *   Plain Python requests to the unified platform (my.gov.sa) were refused (HTTP 403 /
 *   Cloudflare). Dalil does not evade such blocks. mc.gov.sa serves these public pages
 *   normally to a browser; this script only requests the same public pages a person
 *   would open, sequentially, with a delay, and stores the <article> HTML verbatim.
 *   Parsing happens later, in Python (src/ingestion/mc_catalog.py), so it is testable.
 *
 * Progress is stored in the browser's IndexedDB so an interrupted run resumes.
 * Clean-up afterwards:  indexedDB.deleteDatabase('dalil_harvest')
 */
const SERVICE_IDS = [4,91,2,1,6,9,11,33,13,5,67,38,16,46,99,12,41,25,82,19,70,15,66,68,36,28,37,8,17,61,
  20,151,65,3,60,92,150,73,24,43,7,94,52,63,62,18,44,26,97,34,32,42,53,56,40,98,48,152,22,10,27,23,21,
  14,77,76,35,75,55,47,49,31,30,50,51];  // every sID linked from the catalogue page on 2026-09-30

const openDb = () => new Promise((res, rej) => {
  const r = indexedDB.open('dalil_harvest', 1);
  r.onupgradeneeded = () => r.result.createObjectStore('pages', { keyPath: 'key' });
  r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
});
const put = async (rec) => { const db = await openDb(); return new Promise((res, rej) => {
  const tx = db.transaction('pages', 'readwrite'); tx.objectStore('pages').put(rec);
  tx.oncomplete = res; tx.onerror = () => rej(tx.error); }); };
const all = async () => { const db = await openDb(); return new Promise((res) => {
  const q = db.transaction('pages').objectStore('pages').getAll(); q.onsuccess = () => res(q.result); }); };

window.__dalilCapture = async () => {
  const have = new Set((await all()).map(r => r.key));
  for (const id of SERVICE_IDS) for (const lang of ['en', 'ar']) {
    const key = `${lang}-${id}`; if (have.has(key)) continue;
    const path = `/${lang}/eservices/Pages/ServiceDetails.aspx?sID=${id}`;
    const r = await fetch(path, { credentials: 'same-origin' });
    const doc = new DOMParser().parseFromString(await r.text(), 'text/html');
    const art = doc.querySelector('article');
    await put({ key, sid: id, lang, url: 'https://mc.gov.sa' + path, http_status: r.status,
      fetched_at: new Date().toISOString(), page_title: doc.querySelector('title')?.textContent?.trim() || null,
      article_html: art ? art.outerHTML : null });
    await new Promise(res => setTimeout(res, 1500 + Math.random() * 700));   // be polite
  }
  console.log('done');
};

window.__dalilExport = async () => {
  const records = (await all()).sort((a, b) => a.sid - b.sid || a.lang.localeCompare(b.lang));
  const bundle = { source: 'Ministry of Commerce e-services catalogue (mc.gov.sa)',
    method: 'scripts/browser_capture_mc.js (same-origin fetch from a normal browser session, ~2 s spacing)',
    exported_at: new Date().toISOString(), user_agent: navigator.userAgent, records };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(bundle)], { type: 'application/json' }));
  a.download = 'mc_services_capture.json'; a.click();
};

__dalilCapture();
