"""Capture the PUBLIC Absher e-services user guide (Passport + Traffic sectors), EN and AR.

    python -m src.ingestion.capture_absher      (resumable; ~6 s per page, polite pacing)

Plain HTTP GET of public, no-login pages on absher.sa. No login, no CAPTCHA/anti-bot bypass.
Writes data/raw/official/absher/absher_guide_capture.json (gitignored, like the other raw captures):
per page its URL, HTTP status, fetch time, SHA-256 of the full page and the embedded
``currentServiceData`` block that holds the official guide text. Parsed by absher_guide.py.
"""
import json, time, hashlib, urllib.request, os
from datetime import datetime, timezone
from bs4 import BeautifulSoup
UA = "Mozilla/5.0 (Dalil research; official-page capture)"
BASE = "https://www.absher.sa"
from pathlib import Path
OUT = str(Path(__file__).resolve().parents[2] / "data" / "raw" / "official" / "absher" / "absher_guide_capture.json")
def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except Exception as ex:
            err = type(ex).__name__; time.sleep(3 * (k + 1))
    return None, err
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {"source": "absher", "captured_with": "urllib GET of public pages; no login, no anti-bot bypass", "records": []}
done = {(r["lang"], r["sector"], r["id"]) for r in out["records"] if r.get("service_data_js") or r.get("main_html")}
listings = {"en": BASE + "/wps/wcm/connect/individuals/contents/individuals/eServices+User+Guide/eServices/services_listing_page_en",
            "ar": BASE + "/wps/wcm/connect/individuals/contents/individuals+AR/eServices+User+Guide/eServices/services_listing_page_ar"}
for lang, url in listings.items():
    st, h = get(url)
    els = [e for e in BeautifulSoup(h, "html.parser").find_all("a", class_="service-details-link")
           if e.get("data-parent") in ("Passport-Services", "Traffic-Services")]
    print(lang, st, len(els), flush=True)
    for e in els:
        key = (lang, e["data-parent"], e["data-name"])
        if key in done: continue
        u = BASE + e["href"]; time.sleep(0.5)
        st2, page = get(u)
        rec = {"id": e["data-name"], "sector": e["data-parent"], "lang": lang, "url": u, "http_status": st2,
               "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if st2 == 200:
            i = page.find("let currentServiceData"); j = page.find("let relatedServicesIconsData")
            rec.update(sha256=hashlib.sha256(page.encode()).hexdigest(), service_data_js=page[i:j] if i >= 0 else "")
        else:
            rec["error"] = page
        out["records"] = [r for r in out["records"] if (r["lang"], r["sector"], r["id"]) != key] + [rec]
        json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
        n = sum(1 for r in out["records"] if r.get("service_data_js"))
        if n % 10 == 0: print("ok", n, flush=True)
# ---- Absher Business guide (establishment owners: iqama issuance/renewal, exit/re-entry, final exit, ...)
BIZ = BASE + "/wps/portal/business/static/guide/?1dmy&current=true&urile=wcm%3apath%3a%2FBusiness%2BPortal%2FeServices%2BUser%2BGuide%2FElectronic%2BServices%2F"
import re as _re
st, h = get(BIZ)
m = _re.search(r'\[\{"name".*?\}\]', h or "", _re.S)
biz = json.loads(m.group(0)) if m else []
print("business", st, len(biz), flush=True)
def biz_main(page):
    sp = BeautifulSoup(page, "html.parser")
    main = sp.find("div", id="service-title")
    main = main.find_parent("div", class_="col-md-9") if main else None
    return sp, (str(main) if main else "")
for item in biz:
    sid = "biz_" + _re.sub(r"\W+", "_", item["name"]).strip("_").lower()
    for lang in ("en", "ar"):
        key = (lang, "Business-Services", sid)
        if key in done and not (lang == "en" and not next((r.get("ar_url") for r in out["records"]
                                                          if (r["lang"], r["sector"], r["id"]) == key), None)):
            continue
        if lang == "en":
            u = BASE + item["url"]
        else:
            en = next((r for r in out["records"] if (r["lang"], r["sector"], r["id"]) == ("en", "Business-Services", sid)), None)
            u = en.get("ar_url") if en else None
            if not u:
                continue
        time.sleep(0.5)
        st2, page = get(u)
        rec = {"id": sid, "sector": "Business-Services", "lang": lang, "url": u, "http_status": st2,
               "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if st2 == 200:
            sp, main = biz_main(page)
            rec.update(sha256=hashlib.sha256(page.encode()).hexdigest(), main_html=main)
            if lang == "en":
                a = next((x for x in sp.find_all("a", href=True) if x.get_text().strip() == "العربية"), None)
                rec["ar_url"] = (BASE + a["href"]) if a and a["href"].startswith("/") else None
        else:
            rec["error"] = page
        out["records"] = [r for r in out["records"] if (r["lang"], r["sector"], r["id"]) != key] + [rec]
        json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print("DONE", len(out["records"]), sum(1 for r in out["records"] if r.get("service_data_js")), flush=True)
