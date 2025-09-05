import os

# Folder containing the images
folder = r"C:\Arvin\icbt\assignments\Final Project\id_docs_resized\train\passport"
# Get all files and sort them for consistent order
files = sorted(os.listdir(folder))

# Loop and rename in place
for i, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1].lower()  # keep original extension
    new_name = f"image_{i}{ext}"

    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)

print("All files renamed in place to image_1, image_2, ...")
