import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DATA_PATH = Path("data/services.csv")

print("Loading Dalil multilingual AI model...")

model = SentenceTransformer(MODEL_NAME)


def load_services():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/services.csv does not exist yet."
        )

    df = pd.read_csv(DATA_PATH)

    required = [
        "title",
        "agency",
        "category",
        "description",
        "official_url"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.fillna("")

    df["search_text"] = (
        df["title"].astype(str)
        + ". Agency: "
        + df["agency"].astype(str)
        + ". Category: "
        + df["category"].astype(str)
        + ". "
        + df["description"].astype(str)
    )

    return df


def build_index(df):

    print(f"Creating embeddings for {len(df)} official records...")

    embeddings = model.encode(
        df["search_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return np.asarray(embeddings)


def search(query, df, embeddings, top_k=3):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in best_indices:

        row = df.iloc[index]

        results.append({
            "title": row["title"],
            "agency": row["agency"],
            "category": row["category"],
            "description": row["description"],
            "official_url": row["official_url"],
            "score": float(scores[index])
        })

    return results


if __name__ == "__main__":

    df = load_services()
    embeddings = build_index(df)

    print("\n🇸🇦 DALIL AI READY")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("Ask Dalil: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        results = search(
            question,
            df,
            embeddings
        )

        print("\nBEST OFFICIAL MATCHES\n")

        for number, result in enumerate(results, 1):

            print(
                f"{number}. {result['title']}"
            )

            print(
                f"   Agency: {result['agency']}"
            )

            print(
                f"   Similarity: {result['score']:.3f}"
            )

            print(
                f"   {result['description'][:350]}"
            )

            print(
                f"   Source: {result['official_url']}"
            )

            print()

        print("-" * 70)