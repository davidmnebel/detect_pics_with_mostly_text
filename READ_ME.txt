Text-Heavy Image Sorter

This Python script scans a folder of images and moves those that are text-heavy (contain mostly text) into a dedicated subfolder called "text_heavy_images". 
It uses OpenCV and Tesseract OCR for text detection and shows a progress bar during processing.

---

Features

- Detects images containing mostly text.
- Moves text-heavy images to a dedicated folder automatically.
- Supports PNG, JPG, JPEG, BMP, and TIFF images.
- Displays a progress bar while processing.
- Works with Python virtual environments.
- Supports optional command-line arguments for flexibility.

---

Requirements

- Python 3.8 or higher
- Tesseract OCR installed and accessible from your system PATH.

Note: Tesseract must be installed on your system for OCR to work. On macOS, you can install it via "brew install tesseract". 
On Linux, use your package manager. Windows users should download it from the official Tesseract GitHub page and ensure it’s in the PATH.

---

Usage

Run the script:

python main.py --source /path/to/images

- --source : folder containing images to scan. If not provided, the script uses the default folder in main.py.
- --threshold : fraction of image area considered as text to classify as "text-heavy" (default: 0.04).
- --resize : resize factor for faster OCR processing (default: 0.5, in-memory only).

Example:

python main.py --source /Users/david/Pictures/test_images --threshold 0.06 --resize 0.7

---

How It Works

1. Scans all supported images in the source folder.
2. Converts each image to grayscale and resizes it in memory for faster OCR.
3. Uses Tesseract OCR to detect text blocks.
4. Calculates the fraction of the image covered by text.
5. Moves images that meet or exceed the threshold to "text_heavy_images".
6. Shows a progress bar and prints a summary at the end.

---

Notes

- The script does not modify the original images; it only reads them.
- Videos or unsupported file types are ignored.
- For large folders, processing may take some time depending on the number and size of images.

---

Contribution

Feel free to fork the repository, report issues, or submit pull requests to improve the project.