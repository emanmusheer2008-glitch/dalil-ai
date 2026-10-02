"""One Gemini call to check the key/model. Prints only status and Google's error message (never the key).

    python -m src.evaluation.gemini_check
"""
from src.v4.gemini import GeminiClient, GeminiError, api_key

if __name__ == "__main__":
    if not api_key():
        raise SystemExit("GEMINI_API_KEY is not set in this shell.")
    c = GeminiClient()
    k = api_key()
    print("model:", c.model, "| key length:", len(k), "| key has spaces/quotes:", any(x in k for x in " '\"\r\n"))
    try:
        print("OK:", c.generate_json("Return JSON only.", 'Return {"ok": true}', 50))
    except GeminiError as e:
        print("FAILED:", e.code, "|", c.last_error_detail)
