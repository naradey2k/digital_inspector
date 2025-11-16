import time
import json
import os
import numpy as np
import cv2
from pdf2image import convert_from_path
from qreader import QReader
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import supervision as sv
import easyocr
from pathlib import Path

PDFS_DIR = "test"
OUTPUT_JSON = "outputs.json"
OUTPUT_IMAGES_DIR = "annotated_pages"
POPPLER_PATH = r"your_path\poppler-25.11.0\Library\bin"

COLORS = {
    "qr": (0, 0, 255),        
    "signature": (0, 255, 0),  
    "stamp": (255, 0, 0)      
}

print("Loading models...")
qr_model = QReader(model_size='s', min_confidence=0.2)

signature_model_path = hf_hub_download(
    repo_id="tech4humans/yolov8s-signature-detector",
    filename="yolov8s.pt"
)
signature_model = YOLO(signature_model_path)
ocr_reader = easyocr.Reader(['en', 'ru'], gpu=False)

stamp_model = YOLO("best.pt")

print("Models loaded successfully!")

def process_qr_detections(qr_model, image):
    detections = qr_model.detect(image=image)
    annotations = []
    bboxes = []  
    
    for det in detections:
        x1, y1, x2, y2 = map(float, det['bbox_xyxy'])
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        annotations.append({
            "category": "qr",
            "bbox": {
                "x": float(x1),
                "y": float(y1),
                "width": width,
                "height": height
            },
            "area": area
        })
        
        bboxes.append((int(x1), int(y1), int(x2), int(y2), "qr"))
    
    return annotations, bboxes

def process_signature_detections(signature_model, ocr_reader, image):
    image_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    image_height, image_width = image_bgr.shape[:2]
    
    results = signature_model(image_bgr, imgsz=(1024, 768))
    detections = sv.Detections.from_ultralytics(results[0])
    detections = detections[detections.confidence > 0.1]

    annotations = []
    bboxes = []  
    
    for bbox in detections.xyxy:
        x1, y1, x2, y2 = bbox.astype(float)
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        expanded_x1 = 0
        expanded_x2 = image_width
        expanded_y1 = int(y1)
        expanded_y2 = min(int(y2) + 100, image_height)
        
        cropped_region = image_bgr[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
        ocr_results = ocr_reader.readtext(cropped_region)
        extracted_text = ' '.join([result[1] for result in ocr_results])
        
        annotations.append({
            "category": "signature",
            "bbox": {
                "x": float(x1),
                "y": float(y1),
                "width": width,
                "height": height
            },
            "area": area,
            "extracted_text": extracted_text.strip() if extracted_text.strip() else None
        })
        
        bboxes.append((int(x1), int(y1), int(x2), int(y2), "signature"))
    
    return annotations, bboxes

def process_stamp_detections(stamp_model, image):
    image_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    
    results = stamp_model.predict(image_bgr, conf=0.4, imgsz=(876, 1024))
    annotations = []
    bboxes = [] 
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(float)
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            annotations.append({
                "category": "stamp",
                "bbox": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": width,
                    "height": height
                },
                "area": area
            })
            
            bboxes.append((int(x1), int(y1), int(x2), int(y2), "stamp"))
    
    return annotations, bboxes

def draw_annotations(image, all_bboxes):
    annotated_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    
    for x1, y1, x2, y2, category in all_bboxes:
        color = COLORS.get(category, (255, 255, 255))
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 20)
        
        label = category.upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(annotated_image, 
                     (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), 
                     color, -1)
        
        cv2.putText(annotated_image, label, 
                   (x1, y1 - 5), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return annotated_image

def process_pdf(pdf_path, output_images_dir=None):
    print(f"\nProcessing: {pdf_path}")
    
    try:
        pages = convert_from_path(
            pdf_path,
            dpi=300,
            poppler_path=POPPLER_PATH
        )
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        return None
    
    pdf_results = {}
    annotation_counter = 1
    
    pdf_name = os.path.basename(pdf_path)
    pdf_folder_name = os.path.splitext(pdf_name)[0]
    if output_images_dir:
        pdf_output_dir = os.path.join(output_images_dir, pdf_folder_name)
        os.makedirs(pdf_output_dir, exist_ok=True)
    
    for page_num, page in enumerate(pages, start=1):
        print(f"  Processing page {page_num}/{len(pages)}...")
        
        img = np.array(page)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        page_height, page_width = img.shape[:2]
        
        qr_annotations, qr_bboxes = process_qr_detections(qr_model, img)
        signature_annotations, sig_bboxes = process_signature_detections(signature_model, ocr_reader, img)
        stamp_annotations, stamp_bboxes = process_stamp_detections(stamp_model, img)
        
        all_annotations = []
        all_bboxes = qr_bboxes + sig_bboxes + stamp_bboxes
        
        for ann in qr_annotations:
            all_annotations.append({
                f"annotation_{annotation_counter}": ann
            })
            annotation_counter += 1
        
        for ann in signature_annotations:
            all_annotations.append({
                f"annotation_{annotation_counter}": ann
            })
            annotation_counter += 1
        
        for ann in stamp_annotations:
            all_annotations.append({
                f"annotation_{annotation_counter}": ann
            })
            annotation_counter += 1
        
        pdf_results[f"page_{page_num}"] = {
            "annotations": all_annotations,
            "page_size": {
                "width": page_width,
                "height": page_height
            }
        }
        
        if output_images_dir:
            if all_bboxes:
                annotated_image = draw_annotations(img, all_bboxes)
            else:
                annotated_image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
                # pass
            
            output_path = os.path.join(pdf_output_dir, f"page_{page_num}.png")
            cv2.imwrite(output_path, annotated_image)
            print(f"    Saved annotated image: {output_path}")
    
    return pdf_results

def draw_annotations_from_json(json_file_path, output_images_dir):
    print(f"\nDrawing annotations from JSON file: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    pdf_files = {os.path.basename(p): str(p) for p in Path(PDFS_DIR).glob("*.pdf")}
    
    for pdf_name, pdf_data in json_data.items():
        print(f"\nProcessing PDF: {pdf_name}")
        
        pdf_path = pdf_files.get(pdf_name)
        if not pdf_path:
            pdf_path = os.path.join(PDFS_DIR, pdf_name)
            if not os.path.exists(pdf_path):
                print(f"  Warning: PDF file not found: {pdf_name}")
                continue
        
        try:
            pages = convert_from_path(
                pdf_path,
                dpi=300,
                poppler_path=POPPLER_PATH
            )
        except Exception as e:
            print(f"  Error converting PDF {pdf_path}: {e}")
            continue
        
        pdf_folder_name = os.path.splitext(pdf_name)[0]
        pdf_output_dir = os.path.join(output_images_dir, pdf_folder_name)
        os.makedirs(pdf_output_dir, exist_ok=True)
        
        for page_key, page_data in pdf_data.items():
            if not page_key.startswith("page_"):
                continue
            
            page_num = int(page_key.split("_")[1])
            
            if page_num > len(pages):
                print(f"  Warning: Page {page_num} not found in PDF")
                continue
            
            print(f"  Drawing annotations on page {page_num}...")
            
            page = pages[page_num - 1]
            img = np.array(page)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            all_bboxes = []
            annotations = page_data.get("annotations", [])
            
            for ann_obj in annotations:
                for ann_id, ann_data in ann_obj.items():
                    category = ann_data.get("category")
                    bbox = ann_data.get("bbox", {})
                    
                    if category and bbox:
                        x = int(bbox.get("x", 0))
                        y = int(bbox.get("y", 0))
                        width = bbox.get("width", 0)
                        height = bbox.get("height", 0)
                        x2 = x + int(width)
                        y2 = y + int(height)
                        
                        all_bboxes.append((x, y, x2, y2, category))
            
            if all_bboxes:
                annotated_image = draw_annotations(img, all_bboxes)
            else:
                annotated_image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            
            output_path = os.path.join(pdf_output_dir, f"page_{page_num}.png")
            cv2.imwrite(output_path, annotated_image)
            print(f"    Saved annotated image: {output_path}")
    
    print("\nAnnotation drawing complete!")

def main():
    start_time = time.time()
    
    pdf_files = list(Path(PDFS_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDFS_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    
    print("="*50)
    print("Processing all PDFs and generating JSON file with annotations...")
    print("="*50)
    
    all_results = {}
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {os.path.basename(pdf_path)}")
        pdf_results = process_pdf(pdf_path, output_images_dir=OUTPUT_IMAGES_DIR)
        if pdf_results:
            pdf_name = os.path.basename(pdf_path)
            all_results[pdf_name] = pdf_results
    
    json_output_path = os.path.abspath(OUTPUT_JSON)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"\nJSON file with annotations:")
    print(f"  {json_output_path}")
    print(f"\nAnnotated images saved to:")
    print(f"  {os.path.abspath(OUTPUT_IMAGES_DIR)}/")
    print(f"\nProcessed {len(all_results)} PDF file(s)")
    print("="*50)

if __name__ == "__main__":
    main()


