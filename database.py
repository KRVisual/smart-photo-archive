import sqlite3
import exifread
import os

DB_NAME = "photos.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def create_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            filepath TEXT,
            date_taken TEXT,
            width TEXT,
            height TEXT,
            iso TEXT,
            aperture TEXT,
            shutter TEXT,
            focal_length TEXT,
            embedding BLOB
        )
    """)

    conn.commit()
    return conn

def save_photo(conn, photo_data):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO photos (
            filename, filepath, date_taken, width, height,
            iso, aperture, shutter, focal_length, embedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        photo_data["filename"],
        photo_data["filepath"],
        photo_data["date_taken"],
        photo_data["width"],
        photo_data["height"],
        photo_data["iso"],
        photo_data["aperture"],
        photo_data["shutter"],
        photo_data["focal_length"],
        photo_data["embedding"]
    ))
    conn.commit()

def fetch_all_photos():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT filename, filepath, date_taken, width, height,
               iso, aperture, shutter, focal_length, embedding
        FROM photos
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows

def extract_exif(full_path):
    with open(full_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    return {
        "date_taken": str(tags.get("EXIF DateTimeOriginal", "Unknown")),
        "width": str(tags.get("EXIF ExifImageWidth", "Unknown")),
        "height": str(tags.get("EXIF ExifImageLength", "Unknown")),
        "iso": str(tags.get("EXIF ISOSpeedRatings", "Unknown")),
        "aperture": str(tags.get("EXIF FNumber", "Unknown")),
        "shutter": str(tags.get("EXIF ExposureTime", "Unknown")),
        "focal_length": str(tags.get("EXIF FocalLength", "Unknown")),
    }

def scan_and_save(folder_path):
    conn = create_database()

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, filename)

            exif = extract_exif(full_path)

            photo_data = {
                "filename": filename,
                "filepath": full_path,
                "date_taken": exif["date_taken"],
                "width": exif["width"],
                "height": exif["height"],
                "iso": exif["iso"],
                "aperture": exif["aperture"],
                "shutter": exif["shutter"],
                "focal_length": exif["focal_length"],
                "embedding": None
            }

            save_photo(conn, photo_data)
            print(f"✓ Saved metadata: {filename}")

    conn.close()
    print("\nAll photo metadata saved to database!")

if __name__ == "__main__":
    scan_and_save("photos")