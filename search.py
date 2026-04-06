import sqlite3

def search_photos(query):
    conn = sqlite3.connect('photos.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT filename, date_taken, width, height
        FROM photos
        WHERE LOWER(filename) LIKE LOWER(?) OR date_taken LIKE ?
    ''', (f'%{query}%', f'%{query}%'))
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        print("No photos found.")
        return
    
    print(f"\nFound {len(results)} photo(s):\n")
    for row in results:
        print(f"📸 {row[0]}")
        print(f"   Date: {row[1]}")
        print(f"   Size: {row[2]} x {row[3]}")
        print()

query = input("Search your photos: ")
search_photos(query)