import os
import shutil
import cv2
import pytesseract
from pytesseract import Output
from tqdm import tqdm
import argparse
import numpy as np
import string

# ---------------------
# Configurable params
# ---------------------
DEFAULT_SOURCE = "/Users/david/Pictures/random_pics/text_heavy_images"
DEFAULT_THRESHOLD = 0.15    # Fraction of image area considered text-heavy
DEFAULT_RESIZE = 0.5        # Work on a resized copy in memory for speed

# Filters for OCR boxes
CONF_THRESHOLD = 50         # minimum OCR confidence (0-100) to accept a box
MIN_BOX_W = 8               # ignore boxes smaller than this width (pixels, on resized image)
MIN_BOX_H = 6               # ignore boxes smaller than this height (pixels, on resized image)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# ---------------------
# CLI
# ---------------------
parser = argparse.ArgumentParser(description="Move text-heavy images to a folder.")
parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Source folder containing images")
parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Fraction of image area considered text")
parser.add_argument("--resize", type=float, default=DEFAULT_RESIZE, help="Resize factor for faster OCR")
args = parser.parse_args()

SOURCE_FOLDER = args.source
TEXT_THRESHOLD = args.threshold
RESIZE_FACTOR = args.resize
DEST_FOLDER = os.path.join(SOURCE_FOLDER, "text_heavy_images_serious")

os.makedirs(DEST_FOLDER, exist_ok=True)

# ---------------------
# Helper functions
# ---------------------
def preprocess_for_ocr(gray):
    """
    Preprocess grayscale image for better OCR:
    - optional blur to reduce noise
    - Otsu threshold to binarize
    - morphological opening to remove small specks
    Returns the processed image (uint8).
    """
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu threshold (automatic)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological opening to remove small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    return opened

def is_printable_text(s):
    """Return True if s contains at least one printable non-whitespace character."""
    if s is None:
        return False
    txt = str(s).strip()
    if not txt:
        return False
    # ensure there's at least one printable letter/digit/punctuation
    for ch in txt:
        if ch in string.printable and not ch.isspace():
            return True
    return False

def is_text_heavy(image_path, threshold=TEXT_THRESHOLD, resize_factor=RESIZE_FACTOR):
    """
    Returns True if estimated fraction of image area covered by text >= threshold.
    Uses OCR boxes filtered by confidence and size to compute text area.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # resize for faster OCR (work on copy in memory)
    rw = max(1, int(w * resize_factor))
    rh = max(1, int(h * resize_factor))
    resized = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA)

    # preprocess for OCR
    proc = preprocess_for_ocr(resized)

    # Run Tesseract OCR on processed image
    data = pytesseract.image_to_data(proc, output_type=Output.DICT, config="--oem 3 --psm 6")

    total_text_area = 0
    n_boxes = len(data.get('level', []))

    for i in range(n_boxes):
        # text content
        text = data.get('text', [''])[i]
        # confidence: sometimes returns string; try convert to float/int
        conf_raw = data.get('conf', ['-1'])[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0

        # bounding box (on resized image)
        left = int(data.get('left', [0])[i])
        top = int(data.get('top', [0])[i])
        width = int(data.get('width', [0])[i])
        height = int(data.get('height', [0])[i])

        # filter by basic size to ignore tiny noise
        if width < MIN_BOX_W or height < MIN_BOX_H:
            continue

        # filter by confidence
        if conf < CONF_THRESHOLD:
            continue

        # filter by whether text actually contains printable chars
        if not is_printable_text(text):
            continue

        # accept this box
        total_text_area += width * height

    # scale area from resized image back to original image area
    scale_factor = (1 / resize_factor) ** 2
    scaled_text_area = total_text_area * scale_factor
    image_area = float(w * h)
    if image_area <= 0:
        return False

    text_fraction = scaled_text_area / image_area
    # debug print (uncomment for diagnosis)
    # print(f"{os.path.basename(image_path)}: frac={text_fraction:.4f}")

    return text_fraction >= threshold

# ---------------------
# Main processing
# ---------------------
def main():
    # gather image files
    image_files = [f for f in os.listdir(SOURCE_FOLDER)
                   if os.path.isfile(os.path.join(SOURCE_FOLDER, f)) and f.lower().endswith(IMAGE_EXTENSIONS)]

    moved_count = 0

    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        filepath = os.path.join(SOURCE_FOLDER, filename)
        try:
            if is_text_heavy(filepath, threshold=TEXT_THRESHOLD, resize_factor=RESIZE_FACTOR):
                shutil.move(filepath, os.path.join(DEST_FOLDER, filename))
                moved_count += 1
        except Exception as e:
            print(f"Skipped {filename} due to error: {e}")

    # summary
    print("\nProcessing complete!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Text-heavy images moved: {moved_count}")
    print(f"Other images left in folder: {len(image_files) - moved_count}")

if __name__ == "__main__":
    main()