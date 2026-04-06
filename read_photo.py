import exifread
import os

def scan_folder(folder_path):
    photos = []
    
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
            
            photos.append(photo_data)
            print(f"✓ Read: {filename}")
    
    print(f"\nTotal photos scanned: {len(photos)}")
    return photos

scan_folder('photos')