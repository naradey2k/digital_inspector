import time
import numpy as np
import cv2, os
from pdf2image import convert_from_path
from ultralytics import YOLO

def text_coordinates(image):
    results = model.predict(image, conf=0.25, imgsz=(1024, 768))
    coords = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            coords.append(r)
    return coords

model = YOLO("best.pt")

pdf_path = "filename.pdf"
output_dir = "stamp_detected_pages"
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()

pages = convert_from_path(pdf_path, dpi=300,
                        poppler_path=r"your_path\poppler-25.11.0\Library\bin") # path to poppler in windows

for page_num, page in enumerate(pages, start=1):
    img = np.array(page)
    image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)

    coords = text_coordinates(image)
    
    for coord in coords:
        x1, y1, x2, y2 = coord
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 20)  
    
    output_path = os.path.join(output_dir, f"page_{page_num}.png")
    cv2.imwrite(output_path, image)  
    print(f"Saved: {output_path}")
    # Saves drawn bboxes into folder

end_time = time.time()

print(f"Total execution time: {end_time - start_time:.2f} seconds")
