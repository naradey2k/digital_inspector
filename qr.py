import time
import numpy as np
import cv2, os
from pdf2image import convert_from_path
from qreader import QReader

pdf_path = "filename.pdf"
output_dir = "qr_detected_pages"
os.makedirs(output_dir, exist_ok=True)

qr = QReader(model_size='s', min_confidence=0.2)

start_time = time.time()

pages = convert_from_path(pdf_path, dpi=300,
                        poppler_path=r"your_path\poppler-25.11.0\Library\bin") # I needed to install poppler for pdf2image

for page_num, page in enumerate(pages, start=1):
    img = np.array(page)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    detections = qr.detect(image=img)

    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox_xyxy'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 20)

    output_path = os.path.join(output_dir, f"page_{page_num}.png")
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")
    # Creates folder with drawn bounding boxes on pages of document

end_time = time.time()

print(f"Total execution time: {end_time - start_time:.2f} seconds")
