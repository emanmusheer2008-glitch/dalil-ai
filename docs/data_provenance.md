# Data provenance

Dalil V2 indexes **799 services from 10 agencies**. Section A covers the V2 captures; section B is the original V1 Ministry of Commerce capture, unchanged.

## A. V2 agencies (captured 30 Sep – 1 Oct 2026)

| File (`data/raw/official/v2/`) | Agency · site | Pages in file | HTTP status | Services indexed | Notes |
|---|---|---|---|---|---|
| `dalil_v2_hrsd.json` | Human Resources & Social Development · hrsd.gov.sa | 252 | 250×200, 2×404 | 125 | |
| `dalil_v2_zatca.json` | Zakat, Tax and Customs Authority · zatca.gov.sa | 322 | 322×200 | 161 | re-captured 1 Oct |
| `dalil_v2_moe.json` | Ministry of Education · moe.gov.sa | 98 | 97×200, 1 network error | 49 | re-captured 1 Oct; 1 AR page rejected by the language check |
| `dalil_v2_haj.json` | Ministry of Hajj and Umrah · haj.gov.sa | 22 | 22×200 | 11 | re-captured 1 Oct |
| `dalil_v2_momah.json` | Municipalities and Housing · momah.gov.sa | 302 | 302×200 | 150 | re-captured 1 Oct; AR URL from `hreflang`; 2 duplicates removed |
| `dalil_v2_moj.json` | Ministry of Justice · moj.gov.sa | 302 | 302×200 | 148 | 3 duplicates removed |
| `dalil_v2_mofa.json` | Ministry of Foreign Affairs · mofa.gov.sa | 84 | 84×200 | 42 | |
| `dalil_v2_sfda.json` + `dalil_v2_sfda_arfix.json` | Saudi Food and Drug Authority · sfda.gov.sa | 66 + 26 | 40×200, 26×404 → 25 filled by the supplement (1×403) | 25 | 8 pages had no real content (quarantined) |
| `dalil_v2_chi.json` | Council of Health Insurance · chi.gov.sa | 30 | 15×200, 15×404 | 15 | English only (Arabic re-capture not exported) |

**Method.** `scripts/browser_capture_v2.js`, run in a normal Chrome session on each agency's own site: the list of service-detail URLs comes from the agency's own e-services directory; pages are fetched strictly sequentially with a same-origin `fetch`, 1.6–2.4 s apart, no login, no bypassing of any block. Each record stores `url, lang, http_status, fetched_at, page_title` and the page's `<main>` (or `<body>`) HTML **verbatim**. Parsing happens later in Python (`src/ingestion/generic_service_page.py`).

**Arabic URLs.** Taken from each site's own language mapping: the English page's `hreflang="ar"` link (MoMAH, SFDA re-capture), `/en/`→`/ar/` where the site uses it (ZATCA, MoE, Hajj, MOFA, MoJ, HRSD), and CHI's own `switchLanguage()` rule (remove `/en`). Guessing produced 404s and, for MoMAH, a **200 page with the wrong content**, which is why the site's own links are used.

**Supplements** (`dalil_v2_<src>_<tag>.json`) are later re-captures of pages that failed in the main file. The loader lets a supplement *fill* an (id, language) page only when the original page failed; it never replaces a successful original. Both files remain unchanged, and each record lists any supplement pages it used (`extra.supplement_pages`).

**Per record:** agency, official EN/AR URLs, `date_collected`, `last_verified`, SHA-256 of each page's captured HTML, `verification_status = verified_official_capture`.

**What is parsed.** Only *labelled* content: headings, tab labels and their panels, bold or field labels with their values, plus the description paragraph after the title. Labels are mapped to fields with a bilingual keyword table. Not used: related-services cards, FAQ blocks, ratings, feedback widgets, navigation, hidden helpers, unlabelled tab panels, placeholders ("No content available", «لايوجد محتوى متاح») and number-only values whose units are icons.

**Excluded:** 404/403/failed pages, error pages served with HTTP 200, pages whose "Arabic" text isn't Arabic, duplicates (same canonical URL, or the same agency + same English title — the first one is kept; e.g. MoJ lists two pages titled "Notarize divorce"), and records without at least one substantive official field.

## B. V1 source: Ministry of Commerce (unchanged)

| Field | Value |
|---|---|
| Publisher | Ministry of Commerce, Kingdom of Saudi Arabia |
| Pages | `https://mc.gov.sa/{en,ar}/eservices/Pages/ServiceDetails.aspx?sID=<id>` — every service linked from the e-services catalogue page on 2026-09-30 (75 services) |
| Captured | 2026-09-30, 150 pages (75 × English + Arabic), all HTTP 200 |
| Method | `scripts/browser_capture_mc.js`, run in a normal Chrome session on the catalogue page: sequential same-origin requests with a 1.5–2.2 s delay (the server's response time made the real pace slower). The `<article>` element of each page is stored verbatim with URL, timestamp, HTTP status and page title. |
| File | `data/raw/official/mc/mc_services_capture.json` (SHA-256 `5c289b4729a45231d4bfdf0d5f4cd3c742b0df92299201daa86320e9dcfee5a9`) |
| Parser | `src/ingestion/mc_catalog.py` (deterministic, unit-tested) |
| Stored per service | EN/AR URLs, `date_collected`, `last_verified`, SHA-256 of each page's captured HTML, the page's own "Last Modified" date, `verification_status = verified_official_capture` |

### Field mapping (only labelled sections are used)

| Page section (EN / AR) | Schema field |
|---|---|
| service title | `title_*` |
| short labels under the title (e.g. Merchant · Commercial Register · Business sector) | `category_*` (verbatim, joined) |
| description paragraph | `description_*` |
| Conditions / الشروط | `requirements_*` |
| Required Documents / المستندات المطلوبة | `required_documents_*` |
| Steps / الخطوات | `steps_*` |
| Service Fees / رسوم الخدمة | `fees_*` |
| Duration of service / مدة تنفيذ الخدمة | `processing_time_*` |
| Beneficiary Group / الفئة المستفيدة | `target_audience_*` |
| The service is provided in / الخدمة مقدمة باللغة | `service_languages_*` |
| unified number / e-mail | `contact_information` |

`eligibility_*` is left empty: the pages do not have a separate eligibility section. Nothing is translated: `_ar` fields hold the Arabic page's text, `_en` fields the English page's.

### Excluded (quarantined) records — see `data/processed/quarantine.jsonl`

- `mc-53` *Seasonal permits for catering vehicles…* and `mc-56` *Issuing Laboratory License*: the official pages contain only "-" placeholders, so validation rejects them (V2 rule: at least one substantive official field).
- `seed-1` … `seed-18`: the original prototype records. Titles and `my.gov.sa` URLs were entered by hand and descriptions were paraphrased; they could not be re-checked because `my.gov.sa` blocks automated access. They are kept for transparency but never used for answers. Seven of them correspond to Ministry of Commerce services that are now indexed from official pages (e.g. seed-15 *Establish a Limited Liability Company* ↔ `mc-99`).

## Sources that were tried and not used

| Source | Result | Decision |
|---|---|---|
| `my.gov.sa` (unified national platform) | HTTP 403 to Python `requests`, 403 to a hosted fetcher, Cloudflare block page to an automated browser | Stopped contacting it. No bypass attempted. |
| `open.data.gov.sa` (national open-data portal) | Not reachable from the development environment | Adapter built (`official_open_data.py`) for when an official services dataset is downloaded |

## Redistribution

The captured pages are public government information, but no explicit open licence for re-publishing them was checked for each site. Therefore:

- **Raw captures (`data/raw/official/v2/`) are excluded from git** (`.gitignore`). The V1 MC capture was committed in the V1 commit and stays in that history; it is no longer tracked from V2 on.
- **The processed knowledge base** (`data/processed/`) contains verbatim official text and is needed by the deployed app. Before making the repository *public*, check each agency's terms of use (see `docs/DEPLOYMENT.md` → *Data licence decision*). A private repository needs no decision now.

## Refreshing the data

Re-run the capture scripts (`scripts/browser_capture_mc.js`, `scripts/browser_capture_v2.js`), put the JSON files in `data/raw/official/…`, then `python -m src.indexing.build_index && python -m src.evaluation.evaluate_v2`. Embeddings are recomputed only for chunks whose text changed.
