import sqlite3
import numpy as np

conn = sqlite3.connect('photos.db')

# Load all embeddings + filenames
cursor = conn.execute("SELECT filename, embedding FROM photos WHERE embedding IS NOT NULL")
rows = cursor.fetchall()

filenames = [row[0] for row in rows]
embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]

# Find most similar pairs
print("=== VISUALLY SIMILAR PHOTOS ===")
found = False
for i in range(len(embeddings)):
    for j in range(i+1, len(embeddings)):
        similarity = np.dot(embeddings[i], embeddings[j])
        if similarity > 0.85:
            print(f"  {filenames[i]} ↔ {filenames[j]} — similarity: {similarity:.2f}")
            found = True

if not found:
    print("  No similar pairs found above 0.85 threshold")
    print("  Try lowering threshold — here are your top 3 most similar pairs:")
    pairs = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            similarity = np.dot(embeddings[i], embeddings[j])
            pairs.append((similarity, filenames[i], filenames[j]))
    pairs.sort(reverse=True)
    for sim, a, b in pairs[:3]:
        print(f"  {a} ↔ {b} — similarity: {sim:.2f}")

conn.close()