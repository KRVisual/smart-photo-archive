"""
Smart Photo Archive - Streamlit MVP

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.search import search_photos


DATABASE_PATH = PROJECT_ROOT / "photos.db"

CLIP_MODEL = "ViT-B-32"
CLIP_WEIGHTS = "laion2b_s34b_b79k"


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Smart Photo Archive",
    page_icon="📷",
    layout="wide",
)


# ============================================================
# DATABASE
# ============================================================

def connect_database():
    connection = sqlite3.connect(str(DATABASE_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def get_stats():
    stats = {
        "indexed": 0,
        "embedded": 0,
    }

    if not DATABASE_PATH.exists():
        return stats

    connection = connect_database()

    try:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM photos
            WHERE status = 'indexed'
            """
        ).fetchone()

        if row:
            stats["indexed"] = row["total"]

        try:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM photo_embeddings
                """
            ).fetchone()

            if row:
                stats["embedded"] = row["total"]

        except sqlite3.OperationalError:
            pass

    finally:
        connection.close()

    return stats


def get_metadata(photo_id):
    if not DATABASE_PATH.exists():
        return {}

    connection = connect_database()

    try:
        row = connection.execute(
            """
            SELECT *
            FROM photos
            WHERE id = ?
            """,
            (photo_id,),
        ).fetchone()

        if row is None:
            return {}

        return dict(row)

    finally:
        connection.close()


# ============================================================
# SIDEBAR
# ============================================================

stats = get_stats()

with st.sidebar:
    st.title("📷 SPA")

    st.subheader("Archive")

    st.metric(
        "Indexed Photos",
        stats["indexed"],
    )

    st.metric(
        "AI Embeddings",
        stats["embedded"],
    )

    st.divider()

    st.subheader("AI Model")

    st.write(f"**Model:** {CLIP_MODEL}")
    st.write(f"**Weights:** {CLIP_WEIGHTS}")

    st.divider()

    st.subheader("Database")

    st.code(str(DATABASE_PATH))

    st.divider()

    st.caption(
        "🔒 Local-first photo search. "
        "Your photos remain on your computer."
    )


# ============================================================
# HEADER
# ============================================================

st.title("📷 Smart Photo Archive")

st.write(
    "Search your photo library using natural language."
)

st.caption(
    "Local-first AI semantic photo search powered by OpenCLIP."
)


# ============================================================
# STATUS
# ============================================================

if stats["indexed"] > 0 and stats["embedded"] > 0:
    st.success(
        f"Archive ready — {stats['indexed']} photos indexed "
        f"and {stats['embedded']} AI embeddings available."
    )

elif stats["indexed"] > 0:
    st.warning(
        "Photos are indexed, but AI embeddings are missing."
    )

else:
    st.warning(
        "No indexed photos were found."
    )


# ============================================================
# SEARCH UI
# ============================================================

st.subheader("Search Your Photos")

query = st.text_input(
    "Describe what you're looking for",
    placeholder="Try: cherry blossoms",
)

top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=10,
    value=5,
)

search_button = st.button(
    "🔍 Search",
    type="primary",
)


# ============================================================
# SEARCH
# ============================================================

if search_button:

    query = query.strip()

    if not query:
        st.warning(
            "Enter something to search for."
        )

    else:

        with st.spinner(
            f'Searching for "{query}"...'
        ):

            try:
                results = search_photos(
                    query,
                    top_k=top_k,
                )

            except TypeError:
                results = search_photos(
                    query,
                    top_k,
                )

            except Exception as error:
                st.error(
                    "The semantic search failed."
                )

                st.exception(error)
                results = None


        # ====================================================
        # RESULTS
        # ====================================================

        if results is not None:

            if len(results) == 0:

                st.info(
                    "No matching photos found."
                )

            else:

                st.subheader(
                    f'Top matches for "{query}"'
                )

                for rank, result in enumerate(
                    results,
                    start=1,
                ):

                    photo_id = result.photo_id

                    metadata = {}

                    if photo_id is not None:
                        metadata = get_metadata(
                            photo_id
                        )

                    file_name = result.file_name

                    file_path = result.file_path

                    score = result.score

                    width = result.width

                    height = result.height

                    st.divider()

                    st.markdown(
                        f"## #{rank} — {file_name}"
                    )

                    image_column, info_column = (
                        st.columns([2, 1])
                    )


                    # ========================================
                    # IMAGE
                    # ========================================

                    with image_column:

                        if file_path:

                            image_path = Path(
                                file_path
                            )

                            if image_path.exists():

                                try:
                                    st.image(
                                        str(image_path),
                                        use_container_width=True,
                                    )

                                except Exception as error:
                                    st.warning(
                                        "Could not display "
                                        f"this image: {error}"
                                    )

                            else:
                                st.warning(
                                    "Photo file was not found."
                                )

                        else:
                            st.warning(
                                "No photo path available."
                            )


                    # ========================================
                    # INFORMATION
                    # ========================================

                    with info_column:

                        st.metric(
                            "Similarity",
                            f"{score * 100:.1f}%",
                        )

                        if width and height:
                            st.write(
                                f"**Resolution:** "
                                f"{width} × {height}"
                            )

                        date_taken = metadata.get(
                            "date_taken"
                        )

                        if date_taken:
                            st.write(
                                f"**Date taken:** "
                                f"{date_taken}"
                            )

                        camera = metadata.get(
                            "camera_model"
                        )

                        if camera:
                            st.write(
                                f"**Camera:** {camera}"
                            )

                        lens = metadata.get(
                            "lens_model"
                        )

                        if lens:
                            st.write(
                                f"**Lens:** {lens}"
                            )

                        iso = metadata.get(
                            "iso"
                        )

                        if iso is not None:
                            st.write(
                                f"**ISO:** {iso}"
                            )

                        aperture = metadata.get(
                            "aperture"
                        )

                        if aperture is not None:
                            st.write(
                                f"**Aperture:** "
                                f"f/{aperture}"
                            )

                        shutter = metadata.get(
                            "shutter_speed"
                        )

                        if shutter:
                            st.write(
                                f"**Shutter:** "
                                f"{shutter}"
                            )

                        focal_length = metadata.get(
                            "focal_length"
                        )

                        if focal_length is not None:
                            st.write(
                                f"**Focal length:** "
                                f"{focal_length} mm"
                            )

                        if file_path:

                            with st.expander(
                                "File path"
                            ):
                                st.code(
                                    file_path
                                )


# ============================================================
# READY MESSAGE
# ============================================================

elif stats["indexed"] > 0 and stats["embedded"] > 0:

    st.info(
        'Try searching for "cherry blossoms".'
    )


# POWERSHELL SAVE TEST
