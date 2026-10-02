import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import re

BASE = "https://my.gov.sa"
DIRECTORY = "https://my.gov.sa/en/services"

headers = {
    "User-Agent": "Mozilla/5.0 DalilAI-Educational-Project"
}

records = []
seen = set()

print("🇸🇦 Dalil AI — collecting official GOV.SA services...")

for page in range(1, 16):

    url = DIRECTORY if page == 1 else f"{DIRECTORY}?page={page}"

    print(f"Reading directory page {page}...")

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print("Page error:", e)
        continue

    soup = BeautifulSoup(r.text, "html.parser")

    links = soup.find_all("a", href=True)

    service_urls = []

    for link in links:
        href = link["href"]

        if re.match(r"^/en/services/\d+", href):
            full = urljoin(BASE, href)

            if full not in seen:
                seen.add(full)
                service_urls.append(full)

    print(f"Found {len(service_urls)} new service links.")

    for service_url in service_urls:

        try:
            sr = requests.get(
                service_url,
                headers=headers,
                timeout=20
            )
            sr.raise_for_status()

            ssoup = BeautifulSoup(sr.text, "html.parser")

            h1 = ssoup.find("h1")

            if not h1:
                continue

            title = h1.get_text(" ", strip=True)

            text = ssoup.get_text(
                " ",
                strip=True
            )

            # Keep a manageable evidence block for V1
            text = re.sub(r"\s+", " ", text)

            records.append({
                "title": title,
                "content": text[:8000],
                "url": service_url,
                "language": "English",
                "source": "GOV.SA National Platform"
            })

            print(" ✓", title[:65])

            time.sleep(0.25)

        except Exception as e:
            print(" ✗", service_url, e)

df = pd.DataFrame(records)

if not df.empty:
    df = df.drop_duplicates(subset=["url"])

df.to_csv(
    "data/services.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n-----------------------------")
print("DALIL DATA COLLECTION COMPLETE")
print("-----------------------------")
print("Services collected:", len(df))
print("Saved to: data/services.csv")