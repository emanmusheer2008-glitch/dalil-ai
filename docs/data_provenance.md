# Data provenance

## Indexed source

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

- `mc-53` *Seasonal permits for catering vehicles…* and `mc-56` *Issuing Laboratory License*: the official pages have no description (only a "." or "-"), so validation rejects them.
- `seed-1` … `seed-18`: the original prototype records. Titles and `my.gov.sa` URLs were entered by hand and descriptions were paraphrased; they could not be re-checked because `my.gov.sa` blocks automated access. They are kept for transparency but never used for answers. Seven of them correspond to Ministry of Commerce services that are now indexed from official pages (e.g. seed-15 *Establish a Limited Liability Company* ↔ `mc-99`).

## Sources that were tried and not used

| Source | Result | Decision |
|---|---|---|
| `my.gov.sa` (unified national platform) | HTTP 403 to Python `requests`, 403 to a hosted fetcher, Cloudflare block page to an automated browser | Stopped contacting it. No bypass attempted. |
| `open.data.gov.sa` (national open-data portal) | Not reachable from the development environment | Adapter built (`official_open_data.py`) for when an official services dataset is downloaded |

## Redistribution

The captured pages are public government information, but no explicit open licence for re-publishing them was found on the pages. Before making the repository public, check the Ministry of Commerce site terms. If in doubt, keep the repository private (Streamlit Community Cloud can deploy private repos) or publish the code without `data/raw/official/` and rebuild from a fresh capture.

## Refreshing the data

Re-run the capture script (see header of `scripts/browser_capture_mc.js`), replace the JSON, then `python -m src.indexing.build_index && python -m src.evaluation.evaluate`. The cache fingerprint ensures embeddings are recomputed only when text changed.
