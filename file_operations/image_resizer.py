import os
from PIL import Image

input_folder = r"C:\Arvin\icbt\assignments\Final Project\id_docs_original\old_nic"
output_folder = r"C:\Arvin\icbt\assignments\Final Project\id_docs_resized\train\old_nic"

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
rotations = [90, 180, 270]

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            base_name, extension = os.path.splitext(filename)

            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("RGB")

            img_resized = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            img_resized.save(os.path.join(output_folder, filename))

            for angle in rotations:
                img_rotated = img_resized.rotate(angle)
                new_filename = f"{base_name}_rot{angle}{extension}"
                img_rotated.save(os.path.join(output_folder, new_filename))

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"All images resized to {IMG_SIZE} and augmented with rotations.")
print(f"Output saved in: {output_folder}")

