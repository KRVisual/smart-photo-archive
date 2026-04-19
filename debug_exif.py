import exifread

image_path = "photos/IMG_0507.JPG"

with open(image_path, "rb") as f:
    tags = exifread.process_file(f, details=False)

for tag, value in tags.items():
    print(f"{tag}: {value}")