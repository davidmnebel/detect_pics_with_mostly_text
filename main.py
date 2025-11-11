import os
import shutil
import cv2
import pytesseract
from tqdm import tqdm

# --- Configuration ---
SOURCE_FOLDER = "/Users/david/Pictures/random_pics"
DEST_FOLDER = os.path.join(SOURCE_FOLDER, "text_heavy_images")
TEXT_THRESHOLD = 0.04  # Fraction of image area covered by text to consider "mostly text"
RESIZE_FACTOR = 0.5     # Work on half-size copy in memory for speed
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# --- Ensure destination folder exists ---
os.makedirs(DEST_FOLDER, exist_ok=True)

# --- Function to check if image is text-heavy ---
def is_text_heavy(image_path, threshold=TEXT_THRESHOLD):
    image = cv2.imread(image_path)
    if image is None:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize in memory for faster OCR, original image unchanged
    h, w = gray.shape
    resized = cv2.resize(gray, (int(w*RESIZE_FACTOR), int(h*RESIZE_FACTOR)))

    # Use pytesseract with faster config
    data = pytesseract.image_to_data(resized, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6")
    total_text_area = 0
    for i in range(len(data['level'])):
        x, y, width, height = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        total_text_area += width * height

    # Scale text area back to original image size
    scale_factor = (1/RESIZE_FACTOR)**2
    text_fraction = (total_text_area * scale_factor) / (w * h)
    return text_fraction >= threshold

# --- Get list of image files only ---
image_files = [f for f in os.listdir(SOURCE_FOLDER)
               if os.path.isfile(os.path.join(SOURCE_FOLDER, f)) and f.lower().endswith(IMAGE_EXTENSIONS)]

moved_count = 0

# --- Process images with progress bar ---
for filename in tqdm(image_files, desc="Processing images"):
    filepath = os.path.join(SOURCE_FOLDER, filename)
    try:
        if is_text_heavy(filepath):
            shutil.move(filepath, os.path.join(DEST_FOLDER, filename))
            moved_count += 1
    except Exception as e:
        print(f"Skipped {filename} due to error: {e}")

# --- Summary ---
print("\nProcessing complete!")
print(f"Total images processed: {len(image_files)}")
print(f"Text-heavy images moved: {moved_count}")
print(f"Other images left in folder: {len(image_files) - moved_count}")