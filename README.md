# Smart Photo Archive

An AI-powered photo organization system that lets you search your photos
using natural language instead of filenames or folders.

Built in 14 days. Python, CLIP, SQLite, Streamlit.


 What It Does

🔍 Search — Describe a photo in plain English and the AI finds it.
Type "cherry blossoms" and it returns cherry blossom photos.
No tags. No manual work. The AI just understands.

🗂️ Clustering — Automatically groups photos that look visually similar.
Shot the same scene 3 times? It finds all 3 with zero input from you.

⭐ Best Shots — Scores every photo against "professional photography"
and surfaces the best ones automatically. Your own AI photo editor.

---

## How It Works

1. Photos are scanned and EXIF metadata is extracted and stored in SQLite
2. Each photo is passed through OpenAI CLIP to generate a visual embedding
3. Embeddings are saved to the database
4. Search queries are converted to embeddings and matched against photos
5. Clustering groups photos by cosine similarity score between embeddings

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| OpenAI CLIP | Vision-language embeddings |
| SQLite | Database |
| Streamlit | Web UI |
| Pillow | Image processing |

---

## How To Run It

```bash
git clone https://github.com/KRVisual/smart-photo-archive
cd smart-photo-archive
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python database.py
python clip_embed.py
streamlit run visual_search.py
```

Add your photos to the `photos/` folder before running.

---

## What I Learned

- How CLIP converts images into numerical embeddings that capture visual meaning
- How semantic search works — matching meaning not keywords
- How to build and query a SQLite database from scratch
- Why real data is messy — Gmail strips EXIF metadata during transfer
- How to pivot when a technical approach fails
- How to ship something working every single day for 14 days

---

## Built By

Kendall Reed · [@CodesKr1](https://x.com/CodesKr1)  
14 days. Built in public.
