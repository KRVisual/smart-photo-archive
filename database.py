import sqlite3
import exifread
import os

def create_database():
    conn = sqlite3.connect('photos.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            date_taken TEXT,
            width TEXT,
            height TEXT
        )
    ''')
    
    conn.commit()
    return conn

def save_photo(conn, photo_data):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO photos (filename, date_taken, width, height)
        VALUES (?, ?, ?, ?)
    ''', (
        photo_data['filename'],
        photo_data['date_taken'],
        photo_data['width'],
        photo_data['height']
    ))
    conn.commit()

def scan_and_save(folder_path):
    conn = create_database()
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, filename)
            
            with open(full_path, 'rb') as f:
                tags = exifread.process_file(f)
            
            photo_data = {
                'filename': filename,
                'date_taken': str(tags.get('EXIF DateTimeOriginal', 'Unknown')),
                'width': str(tags.get('EXIF ExifImageWidth', 'Unknown')),
                'height': str(tags.get('EXIF ExifImageLength', 'Unknown')),
            }
            
            save_photo(conn, photo_data)
            print(f"✓ Saved: {filename}")
    
    conn.close()
    print("\nAll photos saved to database!")

scan_and_save('photos')