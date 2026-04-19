import sqlite3
import torch
import clip
import numpy as np

DB_NAME = "photos.db"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

def get_connection():
    return sqlite3.connect(DB_NAME)

def get_text_embedding(text):
    text_tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)[0]

def cosine_similarity(a, b):
    return np.dot(a, b)

def search_photos(query, top_k=5):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT filename, filepath, date_taken, width, height,
               iso, aperture, shutter, focal_length, embedding
        FROM photos
        WHERE embedding IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    query_embedding = get_text_embedding(query)
    results = []

    for row in rows:
        filename, filepath, date_taken, width, height, iso, aperture, shutter, focal_length, embedding_blob = row
        image_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        score = cosine_similarity(query_embedding, image_embedding)

        results.append({
            "filename": filename,
            "filepath": filepath,
            "date_taken": date_taken,
            "width": width,
            "height": height,
            "iso": iso,
            "aperture": aperture,
            "shutter": shutter,
            "focal_length": focal_length,
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    query = input("Describe what you're looking for: ")
    matches = search_photos(query)

    print(f"\nTop matches for '{query}':\n")
    for match in matches:
        print(f"📸 {match['filename']} — {match['score'] * 100:.1f}% match")
        print(f"ISO: {match['iso']}")
        print(f"Aperture: {match['aperture']}")
        print(f"Shutter: {match['shutter']}")
        print(f"Focal Length: {match['focal_length']}")
        print(f"Date Taken: {match['date_taken']}")
        print("-" * 40)