import cv2
import os
import supervision as sv
import time
import numpy as np
from pdf2image import convert_from_path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import easyocr

model_path = hf_hub_download(
  repo_id="tech4humans/yolov8s-signature-detector", 
  filename="yolov8s.pt"
)

model = YOLO(model_path)

reader = easyocr.Reader(['en', 'ru'], gpu=False)

pdf_path = "filename.pdf"
output_dir = "sign_detected_pages"
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()

pages = convert_from_path(pdf_path, dpi=300,
                        poppler_path=r"your_path\poppler-25.11.0\Library\bin") # path to poppler

for page_num, page in enumerate(pages, start=1):
    img = np.array(page)
    image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    image_height, image_width = image.shape[:2]

    results = model(image, imgsz=(1024, 768))

    detections = sv.Detections.from_ultralytics(results[0])
    detections = detections[detections.confidence > 0.2]

    print(f"\nPage {page_num}: Found {len(detections)} signature(s)")
    for i, bbox in enumerate(detections.xyxy):
        x1, y1, x2, y2 = bbox.astype(int)
        
        expanded_x1 = 0
        expanded_x2 = image_width
        expanded_y1 = y1
        expanded_y2 = min(y2 + 100, image_height)  # Add 100px to crop line with signature and author
        
        cropped_region = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
        
        results = reader.readtext(cropped_region)
        
        extracted_text = ' '.join([result[1] for result in results])
        
        print(f"\nSignature {i+1} on page {page_num}:")
        print(f"  Original bbox: ({x1}, {y1}, {x2}, {y2})")
        print(f"  Expanded bbox: ({expanded_x1}, {expanded_y1}, {expanded_x2}, {expanded_y2})")
        print(f"  Extracted text:\n{extracted_text.strip()}\n")
        print("-" * 50)

    box_annotator = sv.BoxAnnotator(thickness=20, color=sv.Color.RED)
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    
    output_path = os.path.join(output_dir, f"page_{page_num}.png")
    cv2.imwrite(output_path, annotated_image)  
    print(f"Saved: {output_path}")
    # Saves drawn bboxes into folder

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
