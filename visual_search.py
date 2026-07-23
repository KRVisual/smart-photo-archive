import os
import sqlite3
from collections import Counter
from pathlib import Path

import clip
import exifread
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError


DB_NAME = "photos.db"
DEFAULT_PHOTO_FOLDER = "photos"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

device = "cuda" if torch.cuda.is_available() else "cpu"


st.set_page_config(
    page_title="Smart Photo Archive",
    page_icon="📸",
    layout="wide",
)


@st.cache_resource
def load_clip_model():
    return clip.load("ViT-B/32", device=device)


model, preprocess = load_clip_model()


def get_connection():
    return sqlite3.connect(DB_NAME)


def create_database():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                date_taken TEXT,
                width TEXT,
                height TEXT,
                iso TEXT,
                aperture TEXT,
                shutter TEXT,
                focal_length TEXT,
                embedding BLOB
            )
            """
        )
        conn.commit()


def clean_value(value):
    if value is None:
        return "Not available in this file"

    value = str(value).strip()

    if value == "" or value.lower() == "unknown":
        return "Not available in this file"

    return value


def extract_exif(filepath):
    exif_data = {
        "date_taken": "Unknown",
        "width": "Unknown",
        "height": "Unknown",
        "iso": "Unknown",
        "aperture": "Unknown",
        "shutter": "Unknown",
        "focal_length": "Unknown",
    }

    try:
        with open(filepath, "rb") as file:
            tags = exifread.process_file(file, details=False)

        exif_data["date_taken"] = str(
            tags.get("EXIF DateTimeOriginal", tags.get("Image DateTime", "Unknown"))
        )
        exif_data["iso"] = str(
            tags.get("EXIF ISOSpeedRatings", "Unknown")
        )
        exif_data["aperture"] = str(
            tags.get("EXIF FNumber", "Unknown")
        )
        exif_data["shutter"] = str(
            tags.get("EXIF ExposureTime", "Unknown")
        )
        exif_data["focal_length"] = str(
            tags.get("EXIF FocalLength", "Unknown")
        )

    except Exception:
        pass

    try:
        with Image.open(filepath) as image:
            exif_data["width"] = str(image.width)
            exif_data["height"] = str(image.height)
    except Exception:
        pass

    return exif_data


def get_image_embedding(filepath):
    with Image.open(filepath) as image:
        image = image.convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)

    image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().astype(np.float32)[0]


def get_text_embedding(text):
    text_tokens = clip.tokenize([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy().astype(np.float32)[0]


def find_images(folder_path):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        return []

    image_paths = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(path)

    return sorted(image_paths)


def save_photo(filepath, exif_data, embedding):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO photos (
                filename,
                filepath,
                date_taken,
                width,
                height,
                iso,
                aperture,
                shutter,
                focal_length,
                embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                filename = excluded.filename,
                date_taken = excluded.date_taken,
                width = excluded.width,
                height = excluded.height,
                iso = excluded.iso,
                aperture = excluded.aperture,
                shutter = excluded.shutter,
                focal_length = excluded.focal_length,
                embedding = excluded.embedding
            """,
            (
                os.path.basename(filepath),
                os.path.abspath(filepath),
                exif_data["date_taken"],
                exif_data["width"],
                exif_data["height"],
                exif_data["iso"],
                exif_data["aperture"],
                exif_data["shutter"],
                exif_data["focal_length"],
                embedding.tobytes(),
            ),
        )

        conn.commit()


def index_photo_library(folder_path):
    image_paths = find_images(folder_path)

    if not image_paths:
        st.sidebar.warning("No supported images found in this folder.")
        return

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    successful = 0
    failed = 0

    for index, image_path in enumerate(image_paths, start=1):
        status_text.write(
            f"Indexing {index} of {len(image_paths)}: {image_path.name}"
        )

        try:
            exif_data = extract_exif(str(image_path))
            embedding = get_image_embedding(str(image_path))
            save_photo(str(image_path), exif_data, embedding)
            successful += 1

        except (UnidentifiedImageError, OSError, ValueError) as error:
            failed += 1
            st.sidebar.warning(
                f"Skipped {image_path.name}: {error}"
            )

        except Exception as error:
            failed += 1
            st.sidebar.warning(
                f"Failed to index {image_path.name}: {error}"
            )

        progress_bar.progress(index / len(image_paths))

    status_text.empty()

    st.sidebar.success(
        f"Indexed {successful} photo(s). Failed: {failed}."
    )


def safe_image_path(filepath, selected_folder):
    if not filepath:
        return None

    full_path = os.path.abspath(os.path.normpath(filepath))
    folder_root = os.path.abspath(os.path.normpath(selected_folder))

    try:
        common_path = os.path.commonpath([full_path, folder_root])
    except ValueError:
        return None

    if common_path != folder_root:
        return None

    if not os.path.isfile(full_path):
        return None

    return full_path


def cosine_similarity(vector_a, vector_b):
    return float(np.dot(vector_a, vector_b))


def search_photos(query, top_k=5):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                filename,
                filepath,
                date_taken,
                width,
                height,
                iso,
                aperture,
                shutter,
                focal_length,
                embedding
            FROM photos
            WHERE embedding IS NOT NULL
            """
        ).fetchall()

    if not rows:
        return []

    query_embedding = get_text_embedding(query)
    results = []

    for row in rows:
        (
            filename,
            filepath,
            date_taken,
            width,
            height,
            iso,
            aperture,
            shutter,
            focal_length,
            embedding_blob,
        ) = row

        try:
            image_embedding = np.frombuffer(
                embedding_blob,
                dtype=np.float32,
            )

            if image_embedding.shape != query_embedding.shape:
                continue

            score = cosine_similarity(
                query_embedding,
                image_embedding,
            )

            results.append(
                {
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
                }
            )

        except Exception:
            continue

    results.sort(
        key=lambda photo: photo["score"],
        reverse=True,
    )

    return results[:top_k]


def get_clusters(threshold=0.85):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT filename, filepath, embedding
            FROM photos
            WHERE embedding IS NOT NULL
            """
        ).fetchall()

    valid_rows = []

    for filename, filepath, embedding_blob in rows:
        try:
            embedding = np.frombuffer(
                embedding_blob,
                dtype=np.float32,
            )

            valid_rows.append(
                {
                    "filename": filename,
                    "filepath": filepath,
                    "embedding": embedding,
                }
            )
        except Exception:
            continue

    clusters = []
    assigned = set()

    for first_index in range(len(valid_rows)):
        if first_index in assigned:
            continue

        first_photo = valid_rows[first_index]

        cluster = [
            {
                "filename": first_photo["filename"],
                "filepath": first_photo["filepath"],
            }
        ]

        assigned.add(first_index)

        for second_index in range(
            first_index + 1,
            len(valid_rows),
        ):
            if second_index in assigned:
                continue

            second_photo = valid_rows[second_index]

            if (
                first_photo["embedding"].shape
                != second_photo["embedding"].shape
            ):
                continue

            similarity = cosine_similarity(
                first_photo["embedding"],
                second_photo["embedding"],
            )

            if similarity >= threshold:
                cluster.append(
                    {
                        "filename": second_photo["filename"],
                        "filepath": second_photo["filepath"],
                    }
                )

                assigned.add(second_index)

        clusters.append(cluster)

    return clusters


def get_best_shots(top_k=6):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT filename, filepath, embedding
            FROM photos
            WHERE embedding IS NOT NULL
            """
        ).fetchall()

    if not rows:
        return []

    quality_embedding = get_text_embedding(
        "a sharp professional high quality photograph"
    )

    scored_photos = []

    for filename, filepath, embedding_blob in rows:
        try:
            image_embedding = np.frombuffer(
                embedding_blob,
                dtype=np.float32,
            )

            if image_embedding.shape != quality_embedding.shape:
                continue

            score = cosine_similarity(
                quality_embedding,
                image_embedding,
            )

            scored_photos.append(
                {
                    "filename": filename,
                    "filepath": filepath,
                    "score": score,
                }
            )

        except Exception:
            continue

    scored_photos.sort(
        key=lambda photo: photo["score"],
        reverse=True,
    )

    return scored_photos[:top_k]


def most_common_value(values):
    valid_values = [
        value
        for value in values
        if value != "Not available in this file"
    ]

    if not valid_values:
        return "Not enough data"

    return Counter(valid_values).most_common(1)[0][0]


def analyze_patterns(matches):
    return {
        "iso": most_common_value(
            [match["iso"] for match in matches]
        ),
        "aperture": most_common_value(
            [match["aperture"] for match in matches]
        ),
        "shutter": most_common_value(
            [match["shutter"] for match in matches]
        ),
        "focal_length": most_common_value(
            [match["focal_length"] for match in matches]
        ),
        "date_taken": most_common_value(
            [match["date_taken"] for match in matches]
        ),
    }


def display_photo(photo, selected_folder, show_metadata=False):
    safe_path = safe_image_path(
        photo["filepath"],
        selected_folder,
    )

    if safe_path is None:
        st.error(
            f"Photo cannot be opened: {photo['filename']}"
        )
        return

    try:
        with Image.open(safe_path) as image:
            st.image(
                image.copy(),
                caption=photo.get("caption", photo["filename"]),
                width="stretch",
            )

    except Exception as error:
        st.error(
            f"Could not open {photo['filename']}: {error}"
        )
        return

    if show_metadata:
        st.markdown(
            f"""
**Date Taken:** {photo['date_taken']}  
**Size:** {photo['width']} × {photo['height']}  
**ISO:** {photo['iso']}  
**Aperture:** {photo['aperture']}  
**Shutter:** {photo['shutter']}  
**Focal Length:** {photo['focal_length']}
"""
        )


create_database()


st.sidebar.header("📁 Photo Library")

photo_folder = st.sidebar.text_input(
    "Photo folder path",
    value=DEFAULT_PHOTO_FOLDER,
    help="Enter the full path to a folder containing photos.",
)

folder_exists = os.path.isdir(photo_folder)

if folder_exists:
    image_count = len(find_images(photo_folder))

    st.sidebar.success(
        f"Folder found — {image_count} supported image(s)"
    )
else:
    st.sidebar.warning("Folder not found")

index_button = st.sidebar.button(
    "Index Photo Library",
    disabled=not folder_exists,
    width="stretch",
)

if index_button:
    index_photo_library(photo_folder)
    st.rerun()


st.title("📸 Smart Photo Archive")

st.caption(
    "Search your photos with natural language, discover similar images, "
    "and analyze your photography locally."
)

tab_search, tab_clusters, tab_best = st.tabs(
    [
        "🔍 Search",
        "🗂️ Clusters",
        "⭐ Best Shots",
    ]
)


with tab_search:
    st.subheader("Natural-Language Photo Search")

    st.caption(
        "Try: cherry blossoms, red sunset, bird close up, "
        "waterfall, bridge at night, or photos in Korea."
    )

    query = st.text_input(
        "Describe what you're looking for",
        placeholder="Example: cherry blossoms at night",
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=20,
        value=6,
    )

    if query.strip():
        matches = search_photos(
            query.strip(),
            top_k=top_k,
        )

        st.subheader(
            f"Top matches for “{query.strip()}”"
        )

        if not matches:
            st.warning(
                "No indexed photos were found. "
                "Choose a folder and click Index Photo Library."
            )

        else:
            pattern_summary = analyze_patterns(matches)

            with st.expander(
                "📊 Search Pattern Summary",
                expanded=True,
            ):
                st.markdown(
                    f"""
**Most common ISO:** {pattern_summary['iso']}  
**Most common aperture:** {pattern_summary['aperture']}  
**Most common shutter speed:** {pattern_summary['shutter']}  
**Most common focal length:** {pattern_summary['focal_length']}  
**Most common date taken:** {pattern_summary['date_taken']}
"""
                )

            columns = st.columns(2)

            for index, match in enumerate(matches):
                match["caption"] = (
                    f"{match['filename']} — "
                    f"{match['score'] * 100:.1f}% match"
                )

                with columns[index % 2]:
                    display_photo(
                        match,
                        photo_folder,
                        show_metadata=True,
                    )


with tab_clusters:
    st.subheader("Visually Similar Photo Groups")

    st.caption(
        "Photos are grouped using their CLIP embeddings."
    )

    similarity_threshold = st.slider(
        "Cluster similarity threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.85,
        step=0.01,
    )

    clusters = get_clusters(
        threshold=similarity_threshold
    )

    if not clusters:
        st.info(
            "No indexed photos are available yet."
        )

    for cluster_index, cluster in enumerate(clusters):
        label = (
            f"Cluster {cluster_index + 1} — "
            f"{len(cluster)} photo(s)"
        )

        with st.expander(
            label,
            expanded=len(cluster) > 1,
        ):
            columns = st.columns(
                min(len(cluster), 3)
            )

            for photo_index, photo in enumerate(cluster):
                with columns[photo_index % 3]:
                    display_photo(
                        photo,
                        photo_folder,
                    )


with tab_best:
    st.subheader("AI-Curated Best Shots")

    st.caption(
        "Photos are ranked according to how closely they match "
        "a high-quality professional photography prompt."
    )

    best_shots = get_best_shots(top_k=9)

    if not best_shots:
        st.info(
            "No indexed photos are available yet."
        )

    columns = st.columns(3)

    for index, photo in enumerate(best_shots):
        photo["caption"] = (
            f"{photo['filename']} — "
            f"{photo['score'] * 100:.1f}%"
        )

        with columns[index % 3]:
            display_photo(
                photo,
                photo_folder,
            )