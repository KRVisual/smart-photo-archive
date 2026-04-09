import clip
import torch
from PIL import Image
import os
import sqlite3
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def add_embeddings_column():
    conn = sqlite3.connect('photos.db')
    cursor = conn.cursor()
    try:
        cursor.execute('ALTER TABLE photos ADD COLUMN embedding BLOB')
        conn.commit()
    except:
        pass
    conn.close()

def save_embedding(filename, embedding):
    conn = sqlite3.connect('photos.db')
    cursor = conn.cursor()
    embedding_bytes = pickle.dumps(embedding)
    cursor.execute('UPDATE photos SET embedding = ? WHERE filename = ?', 
                   (embedding_bytes, filename))
    conn.commit()
    conn.close()

def embed_all_photos(folder_path):
    add_embeddings_column()
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, filename)
            
            image = preprocess(Image.open(full_path)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = model.encode_image(image).cpu().numpy()
            
            save_embedding(filename, embedding)
            print(f"✓ Embedded: {filename}")
    
    print("\nAll photos embedded and saved to database!")

embed_all_photos('photos')