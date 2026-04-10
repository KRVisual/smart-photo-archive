import clip
import torch
import sqlite3
import pickle
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def search_by_description(query):
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text).cpu().numpy()

    conn = sqlite3.connect('photos.db')
    cursor = conn.cursor()
    cursor.execute('SELECT filename, embedding FROM photos WHERE embedding IS NOT NULL')
    rows = cursor.fetchall()
    conn.close()

    results = []
    for filename, embedding_bytes in rows:
        photo_embedding = pickle.loads(embedding_bytes)
        similarity = np.dot(text_embedding, photo_embedding.T) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(photo_embedding)
        )
        results.append((filename, float(similarity.flatten()[0])))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop matches for '{query}':\n")
    for filename, score in results[:3]:
        print(f"📸 {filename} — {score:.1%} match")

query = input("Describe what you're looking for: ")
search_by_description(query)