import streamlit as st
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image
import tempfile
import shutil
import sys

# Configuration
ANNOTATED_PAGES_DIR = "annotated_pages"
POPPLER_PATH = r"D:\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"
if not os.path.exists(POPPLER_PATH):
    POPPLER_PATH = None  # Will use system PATH if available

# Suppress print statements during model loading
@st.cache_resource
def load_models():
    """Load models with caching to avoid reloading on every interaction"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Redirect stdout to suppress print statements
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        from qreader import QReader
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        import easyocr
        
        # Load QR model
        qr_model = QReader(model_size='s', min_confidence=0.2)
        
        # Load signature model
        signature_model_path = hf_hub_download(
            repo_id="tech4humans/yolov8s-signature-detector",
            filename="yolov8s.pt"
        )
        signature_model = YOLO(signature_model_path)
        ocr_reader = easyocr.Reader(['en', 'ru'], gpu=False)
        
        # Load stamp model
        stamp_model = YOLO("best.pt")
        
        return qr_model, signature_model, ocr_reader, stamp_model
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

# Import annotation processing functions
try:
    from process_all_models import (
        draw_annotations,
        process_qr_detections,
        process_signature_detections,
        process_stamp_detections,
        COLORS
    )
except ImportError:
    # Define functions locally if import fails
    import supervision as sv
    
    COLORS = {
        "qr": (0, 0, 255),        # Red
        "signature": (0, 255, 0),  # Green
        "stamp": (255, 0, 0)      # Blue
    }
    
    def process_qr_detections(qr_model, image):
        """Detect QR codes in image and return annotations with bbox coordinates"""
        detections = qr_model.detect(image=image)
        annotations = []
        bboxes = []
        
        for det in detections:
            x1, y1, x2, y2 = map(float, det['bbox_xyxy'])
            width = x2 - x1
            height = y2 - y1
            
            annotations.append({
                "category": "qr",
                "bbox": {"x": float(x1), "y": float(y1), "width": width, "height": height},
                "area": width * height
            })
            bboxes.append((int(x1), int(y1), int(x2), int(y2), "qr"))
        
        return annotations, bboxes
    
    def process_signature_detections(signature_model, ocr_reader, image):
        """Detect signatures in image and return annotations with bbox coordinates"""
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
            
            expanded_x1 = 0
            expanded_x2 = image_width
            expanded_y1 = int(y1)
            expanded_y2 = min(int(y2) + 100, image_height)
            
            cropped_region = image_bgr[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
            ocr_results = ocr_reader.readtext(cropped_region)
            extracted_text = ' '.join([result[1] for result in ocr_results])
            
            annotations.append({
                "category": "signature",
                "bbox": {"x": float(x1), "y": float(y1), "width": width, "height": height},
                "area": width * height,
                "extracted_text": extracted_text.strip() if extracted_text.strip() else None
            })
            bboxes.append((int(x1), int(y1), int(x2), int(y2), "signature"))
        
        return annotations, bboxes
    
    def process_stamp_detections(stamp_model, image):
        """Detect stamps in image and return annotations with bbox coordinates"""
        image_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        
        results = stamp_model.predict(image_bgr, conf=0.4, imgsz=(1024, 876))
        annotations = []
        bboxes = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(float)
                width = x2 - x1
                height = y2 - y1
                
                annotations.append({
                    "category": "stamp",
                    "bbox": {"x": float(x1), "y": float(y1), "width": width, "height": height},
                    "area": width * height
                })
                bboxes.append((int(x1), int(y1), int(x2), int(y2), "stamp"))
        
        return annotations, bboxes
    
    def draw_annotations(image, all_bboxes):
        """Draw all bounding boxes on the image with different colors for each category"""
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

st.set_page_config(page_title="PDF Annotation Viewer", layout="wide")

def get_saved_annotations():
    """Get list of all saved annotations from annotated_pages directory"""
    if not os.path.exists(ANNOTATED_PAGES_DIR):
        return {}
    
    saved_annotations = {}
    for folder in os.listdir(ANNOTATED_PAGES_DIR):
        folder_path = os.path.join(ANNOTATED_PAGES_DIR, folder)
        if os.path.isdir(folder_path):
            pages = []
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.png'):
                    pages.append(file)
            if pages:
                saved_annotations[folder] = sorted(pages)
    
    return saved_annotations

def convert_bgr_to_rgb(image):
    """Convert BGR image (from OpenCV) to RGB for display"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_image_for_display(image, max_width=1200):
    """Resize image to a maximum width while maintaining aspect ratio"""
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:
        width, height = image.size
    
    if width <= max_width:
        return image
    
    # Calculate new dimensions
    aspect_ratio = height / width
    new_width = max_width
    new_height = int(new_width * aspect_ratio)
    
    if isinstance(image, np.ndarray):
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    else:
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def process_uploaded_pdf(uploaded_file_bytes, file_name):
    """Process uploaded PDF and return annotated pages"""
    # Load models
    try:
        qr_model, signature_model, ocr_reader, stamp_model = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Convert PDF to images
        pages = convert_from_path(
            tmp_path,
            dpi=300,
            poppler_path=POPPLER_PATH
        )
        
        annotated_pages = []
        
        for page_num, page in enumerate(pages, start=1):
            img = np.array(page)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            # Process with all three models
            qr_annotations, qr_bboxes = process_qr_detections(qr_model, img)
            signature_annotations, sig_bboxes = process_signature_detections(signature_model, ocr_reader, img)
            stamp_annotations, stamp_bboxes = process_stamp_detections(stamp_model, img)
            
            # Combine all bboxes
            all_bboxes = qr_bboxes + sig_bboxes + stamp_bboxes
            
            # Draw annotations
            if all_bboxes:
                annotated_image = draw_annotations(img, all_bboxes)
            else:
                annotated_image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            
            # Convert to RGB for display
            annotated_image_rgb = convert_bgr_to_rgb(annotated_image)
            annotated_pages.append({
                'page_num': page_num,
                'image': annotated_image_rgb,
                'annotations': {
                    'qr': len(qr_annotations),
                    'signature': len(signature_annotations),
                    'stamp': len(stamp_annotations)
                }
            })
        
        return annotated_pages
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def main():
    st.title("📄 PDF Annotation Viewer")
    st.markdown("Upload a PDF to see annotations or view saved annotations from the database")
    
    # Sidebar for saved annotations
    with st.sidebar:
        st.header("📁 Saved Annotations")
        saved_annotations = get_saved_annotations()
        
        if saved_annotations:
            selected_pdf = st.selectbox(
                "Select a PDF to view saved annotations:",
                options=list(saved_annotations.keys()),
                index=0
            )
            
            if selected_pdf:
                st.info(f"📊 {len(saved_annotations[selected_pdf])} pages available")
        else:
            st.warning("No saved annotations found")
            selected_pdf = None
    
    # Main content area
    tab1, tab2 = st.tabs(["📤 Upload PDF", "💾 Saved Annotations"])
    
    # Tab 1: Upload PDF
    with tab1:
        st.header("Upload and Process PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file to detect and display annotations"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            with st.spinner("Processing PDF and detecting annotations... This may take a few moments."):
                file_bytes = uploaded_file.read()
                annotated_pages = process_uploaded_pdf(file_bytes, uploaded_file.name)
            
            if annotated_pages:
                st.success(f"✅ Processed {len(annotated_pages)} page(s)")
                
                # Show summary
                total_qr = sum(p['annotations']['qr'] for p in annotated_pages)
                total_sig = sum(p['annotations']['signature'] for p in annotated_pages)
                total_stamp = sum(p['annotations']['stamp'] for p in annotated_pages)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pages", len(annotated_pages))
                with col2:
                    st.metric("QR Codes", total_qr, delta=None)
                with col3:
                    st.metric("Signatures", total_sig, delta=None)
                with col4:
                    st.metric("Stamps", total_stamp, delta=None)
                
                # Display pages
                st.header("Annotated Pages")
                for page_data in annotated_pages:
                    with st.expander(f"Page {page_data['page_num']} - QR: {page_data['annotations']['qr']}, "
                                   f"Signatures: {page_data['annotations']['signature']}, "
                                   f"Stamps: {page_data['annotations']['stamp']}", expanded=True):
                        # Resize image for better display
                        resized_image = resize_image_for_display(page_data['image'], max_width=1200)
                        st.image(resized_image, use_container_width=False)
                        
                        # Show annotation details
                        if any(page_data['annotations'].values()):
                            st.markdown("**Annotation Summary:**")
                            if page_data['annotations']['qr'] > 0:
                                st.write(f"🔴 QR Codes: {page_data['annotations']['qr']}")
                            if page_data['annotations']['signature'] > 0:
                                st.write(f"🟢 Signatures: {page_data['annotations']['signature']}")
                            if page_data['annotations']['stamp'] > 0:
                                st.write(f"🔵 Stamps: {page_data['annotations']['stamp']}")
            else:
                st.error("Failed to process PDF")
    
    # Tab 2: Saved Annotations
    with tab2:
        st.header("View Saved Annotations")
        
        if saved_annotations:
            if selected_pdf:
                pdf_folder = os.path.join(ANNOTATED_PAGES_DIR, selected_pdf)
                pages = saved_annotations[selected_pdf]
                
                st.subheader(f"📄 {selected_pdf}")
                st.info(f"Total pages: {len(pages)}")
                
                # Display legend
                st.markdown("**Legend:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("🔴 **QR Code**")
                with col2:
                    st.markdown("🟢 **Signature**")
                with col3:
                    st.markdown("🔵 **Stamp**")
                
                # Display all pages
                for page_file in pages:
                    page_path = os.path.join(pdf_folder, page_file)
                    if os.path.exists(page_path):
                        # Extract page number from filename
                        page_num = page_file.replace('page_', '').replace('.png', '')
                        
                        with st.expander(f"Page {page_num}", expanded=False):
                            img = Image.open(page_path)
                            # Resize image for better display
                            resized_img = resize_image_for_display(img, max_width=1200)
                            st.image(resized_img, use_container_width=False)
        else:
            st.warning("No saved annotations available. Process PDFs to generate annotations.")

if __name__ == "__main__":
    main()
