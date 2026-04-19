import sqlite3
import torch
import clip
import numpy as np
from PIL import Image

DB_NAME = "photos.db"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_connection():
    return sqlite3.connect(DB_NAME)

def get_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(image)

    features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)[0]

def update_embeddings():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, filepath FROM photos")
    rows = cursor.fetchall()

    for photo_id, filepath in rows:
        try:
            embedding = get_image_embedding(filepath)
            embedding_bytes = embedding.tobytes()

            cursor.execute("""
                UPDATE photos
                SET embedding = ?
                WHERE id = ?
            """, (embedding_bytes, photo_id))

            print(f"✓ Embedded: {filepath}")

        except Exception as e:
            print(f"✗ Failed: {filepath} -> {e}")

    conn.commit()
    conn.close()
    print("\nAll embeddings saved!")

if __name__ == "__main__":
    update_embeddings()