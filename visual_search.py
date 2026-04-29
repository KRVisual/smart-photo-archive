import os
import sqlite3
from collections import Counter

import numpy as np
import streamlit as st
import torch
import clip
from PIL import Image

DB_NAME = "photos.db"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

st.set_page_config(page_title="Smart Photo Archive", layout="wide")


def get_connection():
    return sqlite3.connect(DB_NAME)


def safe_image_path(filepath):
    photos_root = os.path.abspath("photos")
    full_path = os.path.abspath(os.path.normpath(filepath))
    if not full_path.startswith(photos_root):
        return None
    return full_path


def clean_value(value):
    if value is None:
        return "Not available in this file"
    value = str(value).strip()
    if value == "" or value.lower() == "unknown":
        return "Not available in this file"
    return value


def get_text_embedding(text: str) -> np.ndarray:
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def search_photos(query: str, top_k: int = 5):
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
        (filename, filepath, date_taken, width, height,
         iso, aperture, shutter, focal_length, embedding_blob) = row
        try:
            image_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            score = cosine_similarity(query_embedding, image_embedding)
            results.append({
                "filename": filename,
                "filepath": filepath,
                "date_taken": clean_value(date_taken),
                "width": clean_value(width),
                "height": clean_value(height),
                "iso": clean_value(iso),
                "aperture": clean_value(aperture),
                "shutter": clean_value(shutter),
                "focal_length": clean_value(focal_length),
                "score": score,
            })
        except Exception as e:
            st.warning(f"Could not read embedding for {filename}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def get_clusters(threshold=0.85):
    conn = get_connection()
    cursor = conn.execute(
        "SELECT filename, filepath, embedding FROM photos WHERE embedding IS NOT NULL"
    )
    rows = cursor.fetchall()
    conn.close()

    filenames = [row[0] for row in rows]
    filepaths = [row[1] for row in rows]
    embeddings = [np.frombuffer(row[2], dtype=np.float32) for row in rows]

    clusters = []
    assigned = set()

    for i in range(len(embeddings)):
        if i in assigned:
            continue
        cluster = [{"filename": filenames[i], "filepath": filepaths[i]}]
        assigned.add(i)
        for j in range(i + 1, len(embeddings)):
            if j in assigned:
                continue
            similarity = np.dot(embeddings[i], embeddings[j])
            if similarity > threshold:
                cluster.append({"filename": filenames[j], "filepath": filepaths[j]})
                assigned.add(j)
        clusters.append(cluster)

    return clusters


def get_best_shots(top_k=6):
    conn = get_connection()
    cursor = conn.execute(
        "SELECT filename, filepath, embedding FROM photos WHERE embedding IS NOT NULL"
    )
    rows = cursor.fetchall()
    conn.close()

    query_embedding = get_text_embedding("professional photography high quality")

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


def most_common_value(values):
    valid = [v for v in values if v != "Not available in this file"]
    if not valid:
        return "Not enough data"
    return Counter(valid).most_common(1)[0][0]


def analyze_patterns(matches):
    return {
        "iso": most_common_value([m["iso"] for m in matches]),
        "aperture": most_common_value([m["aperture"] for m in matches]),
        "shutter": most_common_value([m["shutter"] for m in matches]),
        "focal_length": most_common_value([m["focal_length"] for m in matches]),
        "date_taken": most_common_value([m["date_taken"] for m in matches]),
    }


st.title("📸 Smart Photo Archive")
st.caption("Search your photos with natural language and analyze shooting patterns.")

tab1, tab2, tab3 = st.tabs(["🔍 Search", "🗂️ Clusters", "⭐ Best Shots"])

# ── TAB 1: SEARCH ──────────────────────────────────────────────
with tab1:
    st.write("### Try searches like:")
    st.write("- cherry blossoms")
    st.write("- red sunset")
    st.write("- bird close up")
    st.write("- waterfall")
    st.write("- bridge at night")

    query = st.text_input("Describe what you're looking for:")
    top_k = st.slider("Number of results", 1, 10, 5)

    if query:
        matches = search_photos(query, top_k=top_k)
        st.subheader(f"Top matches for '{query}'")

        if not matches:
            st.error("No matches found.")
        else:
            pattern_summary = analyze_patterns(matches)
            st.markdown("## 📊 Pattern Summary")
            st.markdown(f"""
**Most common ISO:** {pattern_summary['iso']}  
**Most common Aperture:** {pattern_summary['aperture']}  
**Most common Shutter:** {pattern_summary['shutter']}  
**Most common Focal Length:** {pattern_summary['focal_length']}  
**Most common Date Taken:** {pattern_summary['date_taken']}
""")
            cols = st.columns(2)
            for i, match in enumerate(matches):
                with cols[i % 2]:
                    safe_path = safe_image_path(match["filepath"])
                    try:
                        if safe_path is None:
                            st.error(f"Invalid path: {match['filepath']}")
                        else:
                            image = Image.open(safe_path)
                            st.image(
                                image,
                                caption=f"{match['filename']} — {match['score'] * 100:.1f}% match",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Could not open image: {match['filepath']}")
                        st.write(str(e))

                    st.markdown(f"""
**Date Taken:** {match['date_taken']}  
**Size:** {match['width']} × {match['height']}  
**ISO:** {match['iso']}  
**Aperture:** {match['aperture']}  
**Shutter:** {match['shutter']}  
**Focal Length:** {match['focal_length']}
""")

# ── TAB 2: CLUSTERS ────────────────────────────────────────────
with tab2:
    st.subheader("🗂️ Visually Similar Photo Groups")
    st.caption("Photos grouped automatically by visual similarity — no manual tagging.")

    clusters = get_clusters(threshold=0.85)

    for idx, cluster in enumerate(clusters):
        label = f"Cluster {chr(65 + idx)} — {len(cluster)} photo{'s' if len(cluster) > 1 else ''}"
        with st.expander(label, expanded=len(cluster) > 1):
            cols = st.columns(min(len(cluster), 3))
            for i, photo in enumerate(cluster):
                with cols[i % 3]:
                    safe_path = safe_image_path(photo["filepath"])
                    try:
                        if safe_path is None:
                            st.error(f"Invalid path: {photo['filename']}")
                        else:
                            image = Image.open(safe_path)
                            st.image(image, caption=photo["filename"], use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not open: {photo['filename']}")

# ── TAB 3: BEST SHOTS ─────────────────────────────────────────
with tab3:
    st.subheader("⭐ AI-Curated Best Shots")
    st.caption("Photos ranked by visual quality — no manual selection.")

    best = get_best_shots(top_k=6)

    cols = st.columns(3)
    for i, photo in enumerate(best):
        with cols[i % 3]:
            safe_path = safe_image_path(photo["filepath"])
            try:
                if safe_path is None:
                    st.error(f"Invalid path: {photo['filename']}")
                else:
                    image = Image.open(safe_path)
                    st.image(
                        image,
                        caption=f"{photo['filename']} — {photo['score']*100:.1f}%",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Could not open: {photo['filename']}")