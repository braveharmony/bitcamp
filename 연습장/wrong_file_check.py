import os
from PIL import Image

def find_corrupted_images(directory):
    corrupted_images = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Bad file {file_path}: {e}")
                corrupted_images.append(file_path)
    return corrupted_images

# Find corrupted images without deleting them
corrupted_images = find_corrupted_images('d:/study_data/_data/cat_dog/PetImages')
print("Corrupted images found:", corrupted_images)