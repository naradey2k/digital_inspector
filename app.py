import streamlit as st
import json
import os
import tempfile
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import cv2
from pdf2image import convert_from_path
from qreader import QReader
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import supervision as sv
import easyocr

# Page configuration
st.set_page_config(
    page_title="Document Annotation Visualizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# File paths
JSON_FILE = "model_outputs.json"
ANNOTATED_IMAGES_DIR = "annotated_pages"
POPPLER_PATH = r"D:\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"

# Colors for different annotation types (BGR format for OpenCV)
COLORS = {
    "qr": (0, 0, 255),        # Red
    "signature": (0, 255, 0),  # Green
    "stamp": (255, 0, 0)      # Blue
}

@st.cache_data
def load_json_data(file_path):
    """Load and cache JSON data"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_models():
    """Load and cache ML models"""
    with st.spinner("Loading models... This may take a few minutes on first run..."):
        # QR model
        qr_model = QReader(model_size='m', min_confidence=0.3)
        
        # Signature model
        signature_model_path = hf_hub_download(
            repo_id="tech4humans/yolov8s-signature-detector",
            filename="yolov8s.pt"
        )
        signature_model = YOLO(signature_model_path)
        
        # OCR reader
        ocr_reader = easyocr.Reader(['en', 'ru'], gpu=False)
        
        # Stamp model
        stamp_model = YOLO("best.pt")
        
    return qr_model, signature_model, ocr_reader, stamp_model

def process_qr_detections(qr_model, image):
    """Detect QR codes in image and return annotations with bbox coordinates"""
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
        area = width * height
        
        # Extract text from expanded region
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
    """Detect stamps in image and return annotations with bbox coordinates"""
    image_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    
    results = stamp_model.predict(image_bgr, conf=0.4, imgsz=(1400, 876))
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

def process_uploaded_pdf(pdf_bytes, pdf_name, qr_model, signature_model, ocr_reader, stamp_model):
    """Process uploaded PDF and return results"""
    # Save uploaded PDF to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Convert PDF to images
        pages = convert_from_path(
            tmp_path,
            dpi=300,
            poppler_path=POPPLER_PATH
        )
    except Exception as e:
        os.unlink(tmp_path)
        raise Exception(f"Error converting PDF: {str(e)}")
    
    pdf_results = {}
    annotation_counter = 1
    annotated_images = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for page_num, page in enumerate(pages, start=1):
        status_text.text(f"Processing page {page_num}/{len(pages)}...")
        progress_bar.progress(page_num / len(pages))
        
        img = np.array(page)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        page_height, page_width = img.shape[:2]
        
        # Process with all three models
        qr_annotations, qr_bboxes = process_qr_detections(qr_model, img)
        signature_annotations, sig_bboxes = process_signature_detections(signature_model, ocr_reader, img)
        stamp_annotations, stamp_bboxes = process_stamp_detections(stamp_model, img)
        
        # Combine all annotations
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
        
        # Draw annotations on image
        if all_bboxes:
            annotated_image = draw_annotations(img, all_bboxes)
        else:
            annotated_image = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        
        # Convert BGR to RGB for display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        annotated_images[page_num] = Image.fromarray(annotated_image_rgb)
    
    progress_bar.empty()
    status_text.empty()
    
    # Clean up temporary file
    os.unlink(tmp_path)
    
    return pdf_results, annotated_images

@st.cache_data
def process_statistics(data):
    """Process JSON data to extract statistics"""
    stats = {
        'total_pdfs': 0,
        'total_pages': 0,
        'total_annotations': 0,
        'by_category': {'qr': 0, 'signature': 0, 'stamp': 0},
        'by_pdf': {},
        'pages_with_annotations': 0,
        'pages_without_annotations': 0,
        'area_stats': {'qr': [], 'signature': [], 'stamp': []}
    }
    
    for pdf_name, pdf_data in data.items():
        stats['total_pdfs'] += 1
        pdf_annotations = {'qr': 0, 'signature': 0, 'stamp': 0}
        
        for page_key, page_data in pdf_data.items():
            if page_key.startswith('page_'):
                stats['total_pages'] += 1
                annotations = page_data.get('annotations', [])
                
                if annotations:
                    stats['pages_with_annotations'] += 1
                else:
                    stats['pages_without_annotations'] += 1
                
                for ann_obj in annotations:
                    for ann_id, ann_data in ann_obj.items():
                        category = ann_data.get('category', 'unknown')
                        if category in stats['by_category']:
                            stats['by_category'][category] += 1
                            pdf_annotations[category] += 1
                            stats['total_annotations'] += 1
                            
                            # Collect area data
                            area = ann_data.get('area', 0)
                            if area > 0:
                                stats['area_stats'][category].append(area)
        
        if any(pdf_annotations.values()):
            stats['by_pdf'][pdf_name] = pdf_annotations
    
    return stats

def get_annotated_image_path(pdf_name, page_num):
    """Get path to annotated image"""
    pdf_folder = os.path.splitext(pdf_name)[0]
    image_path = os.path.join(ANNOTATED_IMAGES_DIR, pdf_folder, f"page_{page_num}.png")
    if os.path.exists(image_path):
        return image_path
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 Document Annotation Visualizer</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_json_data(JSON_FILE)
    
    if not data:
        st.error(f"❌ No data found. Please ensure {JSON_FILE} exists and contains data.")
        st.info("💡 Run `process_all_models.py` to generate the JSON file first.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a view",
        ["📤 Upload PDF", "📊 Overview", "📑 PDF Explorer", "📋 All Annotations", "📈 Statistics", "🖼️ Image Viewer"]
    )
    
    # Process statistics
    stats = process_statistics(data)
    
    # Upload PDF Page
    if page == "📤 Upload PDF":
        st.header("📤 Upload and Process PDF")
        st.markdown("Upload a PDF file to detect QR codes, signatures, and stamps.")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            # Initialize session state for uploaded PDF results
            if 'uploaded_results' not in st.session_state:
                st.session_state.uploaded_results = None
            if 'uploaded_images' not in st.session_state:
                st.session_state.uploaded_images = None
            if 'uploaded_pdf_name' not in st.session_state:
                st.session_state.uploaded_pdf_name = None
            
            # Process button
            if st.button("🔍 Process PDF", type="primary"):
                try:
                    # Load models
                    with st.spinner("Loading models..."):
                        qr_model, signature_model, ocr_reader, stamp_model = load_models()
                    
                    # Process PDF
                    pdf_bytes = uploaded_file.read()
                    pdf_name = uploaded_file.name
                    
                    st.session_state.uploaded_pdf_name = pdf_name
                    pdf_results, annotated_images = process_uploaded_pdf(
                        pdf_bytes, pdf_name, qr_model, signature_model, ocr_reader, stamp_model
                    )
                    
                    st.session_state.uploaded_results = pdf_results
                    st.session_state.uploaded_images = annotated_images
                    
                    st.success("✅ PDF processed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.info("💡 Make sure Poppler is installed and the path is correct.")
            
            # Display results if available
            if st.session_state.uploaded_results is not None:
                st.divider()
                st.subheader("📊 Results")
                
                pdf_results = st.session_state.uploaded_results
                pdf_name = st.session_state.uploaded_pdf_name
                annotated_images = st.session_state.uploaded_images
                
                # Statistics
                total_pages = len([k for k in pdf_results.keys() if k.startswith('page_')])
                total_annotations = sum(
                    len(page_data.get('annotations', []))
                    for page_data in pdf_results.values()
                )
                
                qr_count = sum(
                    1 for page_data in pdf_results.values()
                    for ann_obj in page_data.get('annotations', [])
                    for ann_data in ann_obj.values()
                    if ann_data.get('category') == 'qr'
                )
                
                sig_count = sum(
                    1 for page_data in pdf_results.values()
                    for ann_obj in page_data.get('annotations', [])
                    for ann_data in ann_obj.values()
                    if ann_data.get('category') == 'signature'
                )
                
                stamp_count = sum(
                    1 for page_data in pdf_results.values()
                    for ann_obj in page_data.get('annotations', [])
                    for ann_data in ann_obj.values()
                    if ann_data.get('category') == 'stamp'
                )
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Pages", total_pages)
                with col2:
                    st.metric("Total Annotations", total_annotations)
                with col3:
                    st.metric("QR Codes", qr_count)
                with col4:
                    st.metric("Signatures", sig_count)
                with col5:
                    st.metric("Stamps", stamp_count)
                
                st.divider()
                
                # Page viewer
                st.subheader("📄 View Pages")
                page_nums = sorted([int(k.split('_')[1]) for k in pdf_results.keys() if k.startswith('page_')])
                
                if page_nums:
                    selected_page_num = st.selectbox("Select a page", page_nums)
                    
                    if selected_page_num in annotated_images:
                        # Display annotated image
                        st.image(
                            annotated_images[selected_page_num],
                            caption=f"Page {selected_page_num} - {pdf_name}",
                            use_container_width=True
                        )
                        
                        # Display annotations
                        page_key = f"page_{selected_page_num}"
                        if page_key in pdf_results:
                            page_data = pdf_results[page_key]
                            annotations = page_data.get('annotations', [])
                            
                            if annotations:
                                st.subheader("Annotations on this page")
                                
                                for ann_obj in annotations:
                                    for ann_id, ann_data in ann_obj.items():
                                        category = ann_data.get('category', 'unknown')
                                        bbox = ann_data.get('bbox', {})
                                        area = ann_data.get('area', 0)
                                        
                                        color_map = {
                                            'qr': '🔴',
                                            'signature': '🟢',
                                            'stamp': '🔵'
                                        }
                                        
                                        emoji = color_map.get(category, '⚪')
                                        
                                        with st.expander(f"{emoji} {ann_id} - {category.upper()}"):
                                            st.write(f"**Category:** {category}")
                                            st.write(f"**Position:** ({bbox.get('x', 0):.1f}, {bbox.get('y', 0):.1f})")
                                            st.write(f"**Size:** {bbox.get('width', 0):.1f} × {bbox.get('height', 0):.1f}")
                                            st.write(f"**Area:** {area:.1f} px²")
                                            
                                            if 'extracted_text' in ann_data and ann_data['extracted_text']:
                                                st.write(f"**Extracted Text:** {ann_data['extracted_text']}")
                            else:
                                st.info("No annotations found on this page.")
                
                # Download results as JSON
                st.divider()
                st.subheader("💾 Download Results")
                
                results_json = json.dumps({pdf_name: pdf_results}, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 Download Results as JSON",
                    data=results_json,
                    file_name=f"{os.path.splitext(pdf_name)[0]}_results.json",
                    mime="application/json"
                )
        else:
            st.info("👆 Please upload a PDF file to get started.")
    
    # Overview Page
    elif page == "📊 Overview":
        st.header("📊 Overview Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total PDFs", stats['total_pdfs'])
        with col2:
            st.metric("Total Pages", stats['total_pages'])
        with col3:
            st.metric("Total Annotations", stats['total_annotations'])
        with col4:
            st.metric("Pages with Annotations", stats['pages_with_annotations'])
        
        st.divider()
        
        # Category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Annotations by Category")
            category_df = pd.DataFrame([
                {'Category': 'QR Codes', 'Count': stats['by_category']['qr']},
                {'Category': 'Signatures', 'Count': stats['by_category']['signature']},
                {'Category': 'Stamps', 'Count': stats['by_category']['stamp']}
            ])
            
            fig_pie = px.pie(
                category_df, 
                values='Count', 
                names='Category',
                color_discrete_map={
                    'QR Codes': '#FF6B6B',
                    'Signatures': '#4ECDC4',
                    'Stamps': '#45B7D1'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📈 Category Counts")
            fig_bar = px.bar(
                category_df,
                x='Category',
                y='Count',
                color='Category',
                color_discrete_map={
                    'QR Codes': '#FF6B6B',
                    'Signatures': '#4ECDC4',
                    'Stamps': '#45B7D1'
                }
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Top PDFs by annotation count
        st.subheader("📋 Top PDFs by Annotation Count")
        pdf_counts = []
        for pdf_name, pdf_stats in stats['by_pdf'].items():
            total = sum(pdf_stats.values())
            pdf_counts.append({
                'PDF': pdf_name,
                'Total': total,
                'QR': pdf_stats['qr'],
                'Signature': pdf_stats['signature'],
                'Stamp': pdf_stats['stamp']
            })
        
        if pdf_counts:
            pdf_df = pd.DataFrame(pdf_counts).sort_values('Total', ascending=False).head(10)
            st.dataframe(pdf_df, use_container_width=True, hide_index=True)
    
    # PDF Explorer Page
    elif page == "📑 PDF Explorer":
        st.header("📑 PDF Explorer")
        
        # PDF selector
        pdf_names = list(data.keys())
        selected_pdf = st.selectbox("Select a PDF", pdf_names)
        
        if selected_pdf:
            pdf_data = data[selected_pdf]
            
            # PDF statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_pages = len([k for k in pdf_data.keys() if k.startswith('page_')])
            total_annotations = sum(
                len(page_data.get('annotations', []))
                for page_data in pdf_data.values()
                if isinstance(page_data, dict)
            )
            
            qr_count = sum(
                1 for page_data in pdf_data.values()
                if isinstance(page_data, dict)
                for ann_obj in page_data.get('annotations', [])
                for ann_data in ann_obj.values()
                if ann_data.get('category') == 'qr'
            )
            
            sig_count = sum(
                1 for page_data in pdf_data.values()
                if isinstance(page_data, dict)
                for ann_obj in page_data.get('annotations', [])
                for ann_data in ann_obj.values()
                if ann_data.get('category') == 'signature'
            )
            
            stamp_count = sum(
                1 for page_data in pdf_data.values()
                if isinstance(page_data, dict)
                for ann_obj in page_data.get('annotations', [])
                for ann_data in ann_obj.values()
                if ann_data.get('category') == 'stamp'
            )
            
            with col1:
                st.metric("Total Pages", total_pages)
            with col2:
                st.metric("Total Annotations", total_annotations)
            with col3:
                st.metric("QR Codes", qr_count)
            with col4:
                st.metric("Signatures", sig_count)
            
            st.metric("Stamps", stamp_count)
            
            st.divider()
            
            # Page selector
            page_keys = [k for k in pdf_data.keys() if k.startswith('page_')]
            page_keys.sort(key=lambda x: int(x.split('_')[1]))
            
            selected_page = st.selectbox("Select a page", page_keys)
            
            if selected_page:
                page_data = pdf_data[selected_page]
                page_num = int(selected_page.split('_')[1])
                
                # Page information
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"Page {page_num} Details")
                    
                    # Display annotated image if available
                    image_path = get_annotated_image_path(selected_pdf, page_num)
                    if image_path:
                        img = Image.open(image_path)
                        st.image(img, caption=f"Annotated Page {page_num}", use_container_width=True)
                    else:
                        st.info("⚠️ Annotated image not found. Run the annotation drawing process to generate images.")
                
                with col2:
                    st.subheader("Annotations")
                    annotations = page_data.get('annotations', [])
                    
                    if annotations:
                        for i, ann_obj in enumerate(annotations, 1):
                            for ann_id, ann_data in ann_obj.items():
                                category = ann_data.get('category', 'unknown')
                                bbox = ann_data.get('bbox', {})
                                area = ann_data.get('area', 0)
                                
                                # Color coding
                                color_map = {
                                    'qr': '🔴',
                                    'signature': '🟢',
                                    'stamp': '🔵'
                                }
                                
                                emoji = color_map.get(category, '⚪')
                                
                                with st.expander(f"{emoji} {ann_id} - {category.upper()}"):
                                    st.write(f"**Category:** {category}")
                                    st.write(f"**Position:** ({bbox.get('x', 0):.1f}, {bbox.get('y', 0):.1f})")
                                    st.write(f"**Size:** {bbox.get('width', 0):.1f} × {bbox.get('height', 0):.1f}")
                                    st.write(f"**Area:** {area:.1f} px²")
                                    
                                    if 'extracted_text' in ann_data and ann_data['extracted_text']:
                                        st.write(f"**Extracted Text:** {ann_data['extracted_text']}")
                    else:
                        st.info("No annotations found on this page.")
                    
                    # Page size info
                    page_size = page_data.get('page_size', {})
                    if page_size:
                        st.subheader("Page Size")
                        st.write(f"Width: {page_size.get('width', 0)} px")
                        st.write(f"Height: {page_size.get('height', 0)} px")
    
    # All Annotations Page
    elif page == "📋 All Annotations":
        st.header("📋 All Annotations")
        st.markdown("View and search all predicted annotations from all PDFs.")
        
        # Extract all annotations into a flat list
        all_annotations_list = []
        for pdf_name, pdf_data in data.items():
            for page_key, page_data in pdf_data.items():
                if page_key.startswith('page_'):
                    page_num = int(page_key.split('_')[1])
                    annotations = page_data.get('annotations', [])
                    
                    for ann_obj in annotations:
                        for ann_id, ann_data in ann_obj.items():
                            category = ann_data.get('category', 'unknown')
                            bbox = ann_data.get('bbox', {})
                            area = ann_data.get('area', 0)
                            extracted_text = ann_data.get('extracted_text', '')
                            
                            all_annotations_list.append({
                                'Annotation ID': ann_id,
                                'PDF': pdf_name,
                                'Page': page_num,
                                'Category': category.upper(),
                                'X': f"{bbox.get('x', 0):.1f}",
                                'Y': f"{bbox.get('y', 0):.1f}",
                                'Width': f"{bbox.get('width', 0):.1f}",
                                'Height': f"{bbox.get('height', 0):.1f}",
                                'Area (px²)': f"{area:.1f}",
                                'Extracted Text': extracted_text if extracted_text else 'N/A'
                            })
        
        if not all_annotations_list:
            st.info("No annotations found in the data.")
        else:
            # Filters
            st.subheader("🔍 Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Category filter
                all_categories = sorted(set([ann['Category'] for ann in all_annotations_list]))
                selected_categories = st.multiselect(
                    "Filter by Category",
                    all_categories,
                    default=all_categories
                )
            
            with col2:
                # PDF filter
                all_pdfs = sorted(set([ann['PDF'] for ann in all_annotations_list]))
                selected_pdfs = st.multiselect(
                    "Filter by PDF",
                    all_pdfs,
                    default=all_pdfs
                )
            
            with col3:
                # Search text
                search_text = st.text_input("Search in extracted text", "")
            
            # Apply filters
            filtered_annotations = all_annotations_list.copy()
            
            if selected_categories:
                filtered_annotations = [ann for ann in filtered_annotations if ann['Category'] in selected_categories]
            
            if selected_pdfs:
                filtered_annotations = [ann for ann in filtered_annotations if ann['PDF'] in selected_pdfs]
            
            if search_text:
                filtered_annotations = [
                    ann for ann in filtered_annotations 
                    if search_text.lower() in ann['Extracted Text'].lower()
                ]
            
            # Display statistics
            st.subheader("📊 Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Annotations", len(all_annotations_list))
            with col2:
                st.metric("Filtered Annotations", len(filtered_annotations))
            with col3:
                qr_count = len([ann for ann in filtered_annotations if ann['Category'] == 'QR'])
                st.metric("QR Codes", qr_count)
            with col4:
                sig_count = len([ann for ann in filtered_annotations if ann['Category'] == 'SIGNATURE'])
                st.metric("Signatures", sig_count)
            
            st.metric("Stamps", len([ann for ann in filtered_annotations if ann['Category'] == 'STAMP']))
            
            st.divider()
            
            # Display table
            st.subheader("📋 Annotations Table")
            
            if filtered_annotations:
                # Create DataFrame
                df = pd.DataFrame(filtered_annotations)
                
                # Reorder columns for better display
                column_order = ['Annotation ID', 'PDF', 'Page', 'Category', 'X', 'Y', 'Width', 'Height', 'Area (px²)', 'Extracted Text']
                df = df[column_order]
                
                # Add color coding for categories
                def color_category(val):
                    if val == 'QR':
                        return 'background-color: #ffcccc'
                    elif val == 'SIGNATURE':
                        return 'background-color: #ccffcc'
                    elif val == 'STAMP':
                        return 'background-color: #ccccff'
                    return ''
                
                # Apply styling
                try:
                    styled_df = df.style.applymap(color_category, subset=['Category'])
                except:
                    # If styling fails, use unstyled dataframe
                    styled_df = df
                
                # Display with pagination
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=600,
                    hide_index=True
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Filtered Annotations as CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="annotations.csv",
                    mime="text/csv"
                )
                
                # Detailed view
                st.divider()
                st.subheader("🔍 Detailed View")
                
                if len(filtered_annotations) > 0:
                    selected_idx = st.selectbox(
                        "Select an annotation to view details",
                        range(len(filtered_annotations)),
                        format_func=lambda x: f"{filtered_annotations[x]['Annotation ID']} - {filtered_annotations[x]['PDF']} (Page {filtered_annotations[x]['Page']})"
                    )
                    
                    selected_ann = filtered_annotations[selected_idx]
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write("**Annotation Details:**")
                        st.json({
                            'ID': selected_ann['Annotation ID'],
                            'PDF': selected_ann['PDF'],
                            'Page': selected_ann['Page'],
                            'Category': selected_ann['Category'],
                            'Bounding Box': {
                                'X': selected_ann['X'],
                                'Y': selected_ann['Y'],
                                'Width': selected_ann['Width'],
                                'Height': selected_ann['Height']
                            },
                            'Area': selected_ann['Area (px²)'],
                            'Extracted Text': selected_ann['Extracted Text']
                        })
                    
                    with col2:
                        # Try to show the annotated image
                        pdf_name = selected_ann['PDF']
                        page_num = selected_ann['Page']
                        image_path = get_annotated_image_path(pdf_name, page_num)
                        
                        if image_path:
                            img = Image.open(image_path)
                            st.image(img, caption=f"{pdf_name} - Page {page_num}", use_container_width=True)
                        else:
                            st.info("⚠️ Annotated image not available for this page.")
            else:
                st.warning("No annotations match the selected filters.")
    
    # Statistics Page
    elif page == "📈 Statistics":
        st.header("📈 Detailed Statistics")
        
        # Area distribution
        st.subheader("📏 Annotation Area Distribution")
        
        area_data = []
        for category, areas in stats['area_stats'].items():
            if areas:
                for area in areas:
                    area_data.append({
                        'Category': category.upper(),
                        'Area (px²)': area
                    })
        
        if area_data:
            area_df = pd.DataFrame(area_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_box = px.box(
                    area_df,
                    x='Category',
                    y='Area (px²)',
                    color='Category',
                    color_discrete_map={
                        'QR': '#FF6B6B',
                        'SIGNATURE': '#4ECDC4',
                        'STAMP': '#45B7D1'
                    }
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                fig_violin = px.violin(
                    area_df,
                    x='Category',
                    y='Area (px²)',
                    color='Category',
                    color_discrete_map={
                        'QR': '#FF6B6B',
                        'SIGNATURE': '#4ECDC4',
                        'STAMP': '#45B7D1'
                    }
                )
                st.plotly_chart(fig_violin, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Area Summary Statistics")
            summary = area_df.groupby('Category')['Area (px²)'].describe()
            st.dataframe(summary, use_container_width=True)
        
        # PDF comparison
        st.subheader("📊 PDF Comparison")
        if stats['by_pdf']:
            comparison_data = []
            for pdf_name, pdf_stats in stats['by_pdf'].items():
                comparison_data.append({
                    'PDF': pdf_name,
                    'QR': pdf_stats['qr'],
                    'Signature': pdf_stats['signature'],
                    'Stamp': pdf_stats['stamp']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Stacked bar chart
            fig_stacked = go.Figure()
            
            categories = ['QR', 'Signature', 'Stamp']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for cat, color in zip(categories, colors):
                fig_stacked.add_trace(go.Bar(
                    name=cat,
                    x=comparison_df['PDF'],
                    y=comparison_df[cat],
                    marker_color=color
                ))
            
            fig_stacked.update_layout(
                barmode='stack',
                xaxis_title='PDF',
                yaxis_title='Number of Annotations',
                height=500
            )
            st.plotly_chart(fig_stacked, use_container_width=True)
    
    # Image Viewer Page
    elif page == "🖼️ Image Viewer":
        st.header("🖼️ Image Viewer")
        
        # Get all available PDFs with images
        available_pdfs = []
        if os.path.exists(ANNOTATED_IMAGES_DIR):
            for item in os.listdir(ANNOTATED_IMAGES_DIR):
                item_path = os.path.join(ANNOTATED_IMAGES_DIR, item)
                if os.path.isdir(item_path):
                    # Try to find matching PDF name
                    for pdf_name in data.keys():
                        if os.path.splitext(pdf_name)[0] == item:
                            available_pdfs.append(pdf_name)
                            break
        
        if not available_pdfs:
            st.warning("⚠️ No annotated images found. Run the annotation drawing process first.")
        else:
            selected_pdf = st.selectbox("Select a PDF", available_pdfs)
            
            if selected_pdf:
                pdf_folder = os.path.splitext(selected_pdf)[0]
                image_dir = os.path.join(ANNOTATED_IMAGES_DIR, pdf_folder)
                
                # Get all page images
                page_images = sorted(
                    [f for f in os.listdir(image_dir) if f.endswith('.png')],
                    key=lambda x: int(x.split('_')[1].split('.')[0])
                )
                
                if page_images:
                    # Page selector
                    page_options = [f"Page {int(f.split('_')[1].split('.')[0])}" for f in page_images]
                    selected_page_idx = st.selectbox("Select a page", range(len(page_images)), format_func=lambda x: page_options[x])
                    
                    # Display image
                    image_path = os.path.join(image_dir, page_images[selected_page_idx])
                    img = Image.open(image_path)
                    st.image(img, caption=f"{selected_pdf} - {page_options[selected_page_idx]}", use_container_width=True)
                    
                    # Image info
                    st.info(f"📏 Image size: {img.size[0]} × {img.size[1]} pixels")
                else:
                    st.warning("No images found in this PDF folder.")

if __name__ == "__main__":
    main()

