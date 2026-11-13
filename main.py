import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from data import MENU, TESTS

load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts in a single API call."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in response.data])


items = [(name, description, price) for id, name, description, price in MENU]

# Embed each item as "name: description" text (price carries no meaning for search).
item_texts = [f"{name}: {description}" for name, description, price in items]
items_embedded = embed(item_texts)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


THRESHOLD = 0.50

correct = 0
for test_question, expected_id in TESTS:
    test_embedding = embed([test_question])[0]
    scores = [cosine_similarity(test_embedding, item) for item in items_embedded]
    best_index = int(np.argmax(scores))

    if scores[best_index] < THRESHOLD:
        # Below the threshold: nothing on the menu matches.
        predicted_id = None
    else:
        predicted_id = MENU[best_index][0]

    if predicted_id == expected_id:
        correct += 1

print(f"Correct (threshold {THRESHOLD}): {correct}/{len(TESTS)}")
