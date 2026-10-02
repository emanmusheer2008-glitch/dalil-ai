# Manual acceptance tests

Run in the app (`streamlit run app.py`) or from the command line (`python -m src.engine "<question>"`). None of these questions is special-cased in the code. The "Result" column shows what the final system actually returned on 1 Oct 2026 (`decision · top score · first service`), so you can see whether behaviour changes later.

Decision thresholds: decline < 0.285 ≤ possible match < 0.391 ≤ answer.

| # | Requirement | Question | Expected | Result (1 Oct 2026) | Pass? |
|---|---|---|---|---|---|
| A | Known English test | How can I reserve a trade name? | Trade Name Reservation, not refused | possible match · 0.382 · Trade Name Reservation (mc-1) | ✅ (shown with a "possible match" badge) |
| B | Paraphrased English | I want to book a business name for my company. | Trade Name Reservation | answered · 0.436 · Trade Name Reservation (mc-1) | ✅ |
| C | Arabic equivalent | كيف أحجز اسماً تجارياً لشركتي؟ | حجز اسم تجاري | possible match · 0.389 · حجز اسم تجاري (mc-1) | ✅ (refused before the tanween fix, see DEVELOPMENT_LOG Issue 11) |
| D | English, different agency | How do I apply for a family visit visa for my parents? | Family Visit Visa (MOFA) | answered · 0.489 · Family Visit Visa | ✅ |
| E | Arabic, different agency | كيف أستعلم عن صك عقاري؟ | Real Estate Title Deed Inquiry (MoJ) | declined · 0.257 | ❌ **false refusal**: the verb form «أستعلم» doesn't match the noun «الاستعلام» |
| F | Mixed language | كم رسوم VAT registration للمؤسسة؟ | VAT Registration for Businesses (fees first) | possible match · 0.373 · *VAT Registration Verification* (sibling), fees section first | ⚠️ partly: right topic and fees first, but a sibling service |
| G | Unsupported / out of scope | How do I renew my driving licence? | decline (traffic/MOI not indexed) | possible match · 0.310 · *Renew license* (MC professional licence) | ❌ shown as a possible match instead of declined (known limitation: nearby missing services) |
| H | Ambiguous | I want to file a complaint | several candidates, low confidence | possible match · 0.385 · File complaint against an employer (CHI), 3 sources | ⚠️ acceptable: flagged as uncertain, but Dalil doesn't ask a clarifying question |
| I | Needs several passages | What documents and fees are needed for a building permit? | Issuing a Building Permit, with documents/requirements and fees sections | answered · 0.419 · Issuing a Building Permit; sections: fees first, requirements, steps; 3 sources | ✅ |
| J | Natural, not a title | My worker ran away, what should I do? | decline: no absence/«تغيب» report service is in the indexed HRSD pages (searched titles in EN + AR) | declined · 0.210 | ✅ correct decline (no such service in the corpus) |

**Summary:** 6 pass (A, B, C, D, I, J), 2 partial (F, H), 2 fail (E, G). The failures are listed in the README limitations, not hidden.

**Extra checks for the app**
- Switch to العربية: the whole interface should be right-to-left, and Arabic typed into the box aligns right.
- Narrow the browser to phone width: cards stack, with no horizontal scrolling.
- Every answer shows an official link, an agency and a capture date. Click one link.
- Out-of-scope examples («طريقة عمل الكبسة», "What is the best pizza in Riyadh?") show the "not enough verified information" card.
