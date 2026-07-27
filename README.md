# 📷 Smart Photo Archive

An AI-powered, local-first photo organization system that lets you search and explore your own photos using artificial intelligence instead of relying only on filenames or folders.

**Built in 14 days. Python, CLIP / OpenCLIP, SQLite, Streamlit.**

---

## What It Does

### 🔍 Semantic Search

Describe a photo in plain English and Smart Photo Archive finds the most visually relevant matches.

For example:

```text
cherry blossoms
```

SPA can return photos containing cherry blossoms even if the filename contains nothing about flowers.

No manual tags are required. CLIP embeddings allow the system to compare the meaning of a text query with the visual content of your photos.

### 🗂️ Clustering

Smart Photo Archive can automatically group photos that are visually similar.

Shot the same scene multiple times?

SPA can compare image embeddings and identify photographs with similar visual content.

### ⭐ Best Shots

Smart Photo Archive can use CLIP similarity to score photographs against a professional-photography prompt and surface stronger candidates from the library.

This provides a simple AI-assisted way to explore potential best shots.

### 📁 Use Your Own Photos

Smart Photo Archive is designed to work with your own photo library.

Point SPA at a folder on your computer and the indexer recursively discovers supported photographs inside that folder and its subfolders.

Your original photographs remain in their existing locations.

### ⚡ Incremental Indexing

SPA calculates a **SHA-256 hash** for each photograph.

Previously indexed photos are detected and skipped instead of being processed repeatedly.

During development, 23 real photographs were indexed successfully.

Running the indexer a second time correctly skipped all 23.

### 📸 EXIF & Metadata

Smart Photo Archive extracts and stores available image metadata, including:

- Resolution
- Date taken
- Camera make/model
- Lens model
- ISO
- Aperture
- Shutter speed
- Focal length
- GPS information when available
- Image format

Metadata availability depends on the original photograph.

### 🔒 Local-First

Smart Photo Archive is designed around local processing.

Your original photographs stay on your computer.

Photo records, metadata, hashes, and AI embeddings are stored locally using SQLite.

---

# Demo

## Semantic Search

Example query:

```text
cherry blossoms
```

During MVP testing, Smart Photo Archive returned an actual cherry blossom photograph as the **#1 ranked result**.

The result included:

- Visual preview
- Filename
- Similarity score
- Resolution
- Date taken
- File path

> Add screenshot here showing the `cherry blossoms` search result.

---

# How It Works

```text
Your Photo Library
        ↓
Photo Discovery
        ↓
SHA-256 Hashing
        ↓
EXIF / Metadata Extraction
        ↓
SQLite Photo Index
        ↓
CLIP / OpenCLIP Image Embeddings
        ↓
AI Features
   ┌────┼─────────┐
   ↓    ↓         ↓
 Search Clusters Best Shots
   ↓
Streamlit UI
```

### Semantic Search Pipeline

```text
Natural-Language Query
        ↓
CLIP Text Embedding
        ↓
Compare Against Image Embeddings
        ↓
Cosine Similarity
        ↓
Rank Results
        ↓
Top Matching Photos
```

The current modular pipeline uses OpenCLIP for image and text embeddings.

---

# Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core application |
| CLIP / OpenCLIP | Vision-language embeddings |
| PyTorch | AI model inference |
| SQLite | Local database |
| Streamlit | User interface |
| Pillow | Image processing and EXIF |
| NumPy | Vector and embedding operations |

---

# Project Structure

The newer modular architecture separates indexing, AI, search, and the user interface:

```text
smart-photo-archive/
│
├── ai/
│   ├── clip_embed.py
│   └── search.py
│
├── core/
│   ├── config.py
│   ├── hashing.py
│   ├── indexer.py
│   └── metadata.py
│
├── ui/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

Some earlier project scripts may remain in the repository as part of the original 14-day implementation.

---

# How To Run It

## 1. Clone the Repository

```bash
git clone https://github.com/KRVisual/smart-photo-archive
cd smart-photo-archive
```

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Launch Smart Photo Archive

```powershell
python -m streamlit run .\ui\app.py
```

Streamlit will normally open the application at:

```text
http://localhost:8501
```

---

# Using Your Own Photos

Launch Smart Photo Archive and enter the path to the folder containing your photographs.

For example:

```text
C:\Users\YourName\Pictures
```

Select:

```text
Index My Photos
```

Smart Photo Archive can then:

1. Recursively discover supported images
2. Calculate SHA-256 hashes
3. Skip photographs already indexed
4. Extract available EXIF metadata
5. Store photo information in SQLite
6. Generate missing CLIP embeddings
7. Make the photographs available to the AI search system

The first embedding run may take some time depending on the number of photographs and your hardware.

Later runs can skip photographs and embeddings that have already been processed.

---

# Supported Image Formats

The current indexer supports:

```text
.jpg
.jpeg
.png
.webp
.tif
.tiff
.bmp
```

---

# MVP Validation

Smart Photo Archive was tested using **23 real photographs**.

## Initial Index

```text
23 photos indexed successfully
23 / 23 CLIP embeddings generated
```

## Incremental Index Test

The same library was indexed again.

SPA correctly detected the existing photographs:

```text
23 discovered
0 new
23 skipped
```

## Semantic Search Test

Query:

```text
cherry blossoms
```

Smart Photo Archive returned:

```text
#1 — IMG_0950.JPG
```

The #1 photograph contained cherry blossoms.

This demonstrated that the retrieval system was matching the visual meaning of the photograph rather than depending on its filename.

---

# What I Learned

Building Smart Photo Archive taught me:

- How CLIP represents images and text as numerical embeddings
- How semantic search can match meaning instead of keywords
- How cosine similarity can rank image/text relationships
- How to build an incremental photo indexing pipeline
- How SHA-256 hashing can identify previously processed photographs
- How to extract and work with real EXIF metadata
- How to store photo information and embeddings in SQLite
- How real-world image metadata can be incomplete or stripped during transfer
- How to structure Python code into separate indexing, AI, search, and UI modules
- How to build a local-first AI application
- How to iterate when a technical approach fails
- How to take an AI project from an idea to a working MVP

---

# Current Status

## Smart Photo Archive v1.0 — Portfolio MVP

The core system works end-to-end:

```text
Discover
   ↓
Hash
   ↓
Extract Metadata
   ↓
Index
   ↓
Generate Embeddings
   ↓
Search / Analyze
   ↓
Display Results
```

The goal of Smart Photo Archive is to demonstrate a practical, local AI photo-intelligence system rather than a production-scale cloud photo service.

---

# Built By

**Kendall **

Built in **14 days** and developed in public.

X: [@CodesKr1](https://x.com/CodesKr1)

GitHub: [KRVisual](https://github.com/KRVisual)
