import sqlite3
import numpy as np

conn = sqlite3.connect('photos.db')

cursor = conn.execute("SELECT filename, embedding FROM photos WHERE embedding IS NOT NULL")
rows = cursor.fetchall()

filenames = [row[0] for row in rows]
embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]

# Group into clusters
clusters = []
assigned = set()

for i in range(len(embeddings)):
    if i in assigned:
        continue
    cluster = [filenames[i]]
    assigned.add(i)
    for j in range(i+1, len(embeddings)):
        if j in assigned:
            continue
        similarity = np.dot(embeddings[i], embeddings[j])
        if similarity > 0.85:
            cluster.append(filenames[j])
            assigned.add(j)
    clusters.append(cluster)

# Print clusters
print("=== PHOTO CLUSTERS ===\n")
for idx, cluster in enumerate(clusters):
    label = f"Cluster {chr(65+idx)}"
    print(f"{label} ({len(cluster)} photos):")
    for photo in cluster:
        print(f"  - {photo}")
    print()

print(f"Total: {len(clusters)} clusters from {len(filenames)} photos")

conn.close()