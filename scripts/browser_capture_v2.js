/*
 * Dalil AI — generic capture of official service pages (V2 agencies).
 *
 * Used for: hrsd, zatca, moe, haj, sfda, momah, moj, mofa, chi (V2 corpus).
 * The 1 Oct 2026 re-capture of zatca, moe, haj and momah used this script.
 *
 * WHERE THE SERVICE LISTS CAME FROM (each agency's own directory)
 *   moe   : /en/knowledgecenter/eservices/DataSources/ServicesList.aspx  (49 detail pages)
 *   zatca : /en/eServices/Pages/default.aspx, client-side pager, "Learn More" links (161)
 *   momah : /en/e-services?page=N (Drupal listing, 152); Arabic URL = the English page's
 *           <link rel="alternate" hreflang="ar">  (set it.ar = null to enable this, see below)
 *   haj   : /en/E-Services "Service Details" links (11)
 *   Arabic URL otherwise: the site's own /en/ -> /ar/ pattern. Never guess: SFDA uses Arabic
 *   slugs, CHI drops the language prefix, MoMAH serves wrong content on a guessed URL.
 *
 * DOWNLOAD NOTE: some sites intercept link clicks (ZATCA "Leaving site" pop-up); if the file
 * does not download, dispatch a non-bubbling click instead of a.click():
 *   a.dispatchEvent(new MouseEvent('click', {bubbles: false}))
 *
 * HOW IT IS USED (normal Chrome session, on the agency's own site)
 *   1. Open the agency's e-services directory page.
 *   2. Build `window.__items = [{id, en, ar}, ...]` from the links on that page
 *      (service-detail URLs in English and their Arabic counterparts).
 *   3. Paste this file in DevTools → Console, then run  __dalilCapture('<source>')
 *   4. When it logs "done", run  __dalilExport('<source>')  → dalil_v2_<source>.json
 *
 * POLITENESS / ETHICS
 *   - Same-origin fetch of public pages a person can open; no login, no API keys.
 *   - Strictly sequential, ~2 s between requests (1.6–2.4 s jitter).
 *   - No bypassing of anti-bot checks: a blocked/failed page is recorded with its
 *     HTTP status and later quarantined by the Python pipeline.
 *   - `main_html` is the page's <main> element (or <body> if there is none), stored
 *     verbatim. Nothing is rewritten or translated here; parsing happens in Python.
 *
 * Progress lives in IndexedDB ('dalil_harvest_v2'), so an interrupted run resumes.
 * Clean-up afterwards:  indexedDB.deleteDatabase('dalil_harvest_v2')
 */
(() => {
  const DB = 'dalil_harvest_v2';
  const openDb = () => new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore('pages', { keyPath: 'key' });
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
  const put = async (rec) => { const db = await openDb(); return new Promise((res, rej) => {
    const tx = db.transaction('pages', 'readwrite'); tx.objectStore('pages').put(rec);
    tx.oncomplete = res; tx.onerror = () => rej(tx.error); }); };
  const all = async () => { const db = await openDb(); return new Promise((res) => {
    const q = db.transaction('pages').objectStore('pages').getAll(); q.onsuccess = () => res(q.result); }); };
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  window.__dalilState = { done: 0, total: 0, errors: 0, running: false };

  window.__dalilCapture = async (source) => {
    const items = window.__items || [];
    const have = new Set((await all()).map(r => r.key));
    const st = window.__dalilState; st.total = items.length * 2; st.running = true;
    for (const it of items) for (const lang of ['en', 'ar']) {
      const key = `${source}|${lang}|${it.id}`;
      if (have.has(key)) { st.done++; continue; }
      const url = it[lang];
      if (!url) { st.done++; continue; }                  // no official Arabic page found
      let rec = { key, source, id: it.id, lang, url, http_status: null,
                  fetched_at: new Date().toISOString(), page_title: null, main_html: null };
      try {
        const r = await fetch(url, { credentials: 'same-origin' });
        rec.http_status = r.status;
        const doc = new DOMParser().parseFromString(await r.text(), 'text/html');
        rec.page_title = doc.querySelector('title')?.textContent?.trim() || null;
        if (lang === 'en' && it.ar === null) {            // take the official Arabic URL from the page
          const l = doc.querySelector('link[rel="alternate"][hreflang="ar"]');
          if (l) { it.ar = new URL(l.getAttribute('href'), location.origin).href; rec.alt_ar = it.ar; }
        }
        const main = doc.querySelector('main') || doc.body;
        rec.main_html = main ? main.outerHTML : null;
      } catch (e) { rec.error = String(e); st.errors++; }
      await put(rec); st.done++;
      await sleep(1600 + Math.random() * 800);   // be polite
    }
    st.running = false; console.log('done', source);
  };

  window.__dalilCount = async () => (await all()).length;

  window.__dalilExport = async (source) => {
    const records = (await all()).filter(r => r.source === source)
      .sort((a, b) => String(a.id).localeCompare(String(b.id)) || a.lang.localeCompare(b.lang));
    const bundle = { source, origin: location.origin,
      method: 'scripts/browser_capture_v2.js (same-origin fetch from a normal browser session, ~2 s spacing)',
      exported_at: new Date().toISOString(), user_agent: navigator.userAgent, records };
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(bundle)], { type: 'application/json' }));
    a.download = `dalil_v2_${source}.json`;
    document.documentElement.appendChild(a); a.click(); a.remove();
    return records.length;
  };
})();
