import os
import sqlite3
import numpy as np
import shutil
from PIL import Image

DB_NAME = "photos.db"
OUTPUT_DIR = "portfolio"

def get_connection():
    return sqlite3.connect(DB_NAME)

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def get_best_shots(top_k=20):
    import torch
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    text_tokens = clip.tokenize(["professional photography high quality"]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    query_embedding = text_features.cpu().numpy().astype(np.float32)[0]

    conn = get_connection()
    cursor = conn.execute(
        "SELECT filename, filepath, embedding FROM photos WHERE embedding IS NOT NULL"
    )
    rows = cursor.fetchall()
    conn.close()

    scored = []
    for row in rows:
        filename, filepath, embedding_blob = row
        try:
            image_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            score = cosine_similarity(query_embedding, image_embedding)
            scored.append({"filename": filename, "filepath": filepath, "score": score})
        except:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/photos", exist_ok=True)

print("Finding best shots...")
best = get_best_shots(top_k=20)

# Copy photos to portfolio folder
for photo in best:
    src = os.path.normpath(photo["filepath"])
    dst = f"{OUTPUT_DIR}/photos/{photo['filename']}"
    shutil.copy2(src, dst)

# Build HTML
cards = ""
for photo in best:
    cards += f"""
    <div class="card">
        <img src="photos/{photo['filename']}" alt="{photo['filename']}">
        <p>{photo['score']*100:.1f}% match</p>
    </div>
    """

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ken Reed — Photography Portfolio</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #0a0a0a; color: #fff; font-family: -apple-system, sans-serif; }}
        header {{ padding: 60px 40px; text-align: center; }}
        header h1 {{ font-size: 2.5rem; font-weight: 300; letter-spacing: 4px; }}
        header p {{ color: #888; margin-top: 10px; letter-spacing: 2px; font-size: 0.9rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 4px; padding: 0 4px; }}
        .card img {{ width: 100%; height: 300px; object-fit: cover; display: block; transition: opacity 0.3s; }}
        .card img:hover {{ opacity: 0.8; }}
        .card p {{ display: none; }}
        footer {{ text-align: center; padding: 40px; color: #444; font-size: 0.8rem; letter-spacing: 2px; }}
    </style>
</head>
<body>
    <header>
        <h1>KEN REED</h1>
        <p>AI-CURATED PHOTOGRAPHY — SELECTED BY COMPUTER VISION</p>
    </header>
    <div class="grid">
        {cards}
    </div>
    <footer>
        <p>Photos selected automatically using CLIP vision embeddings · @CodesKr1</p>
    </footer>
</body>
</html>"""

with open(f"{OUTPUT_DIR}/index.html", "w") as f:
    f.write(html)

print(f"Portfolio exported to {OUTPUT_DIR}/index.html")
print(f"Open it in your browser to see it")