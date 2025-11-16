# Document Annotation System

## Features

- **QR Code Detection**: Detects QR codes in PDF documents using QReader
- **Signature Detection**: Identifies signatures using YOLOv8 model with OCR text extraction
- **Stamp Detection**: Detects stamps using a custom-trained YOLO model
- **Web Visualization**: Interactive Streamlit dashboard for viewing and analyzing results

## Technologies Used

### Machine Learning & Computer Vision
- **QReader**: QR code detection and decoding
- **Ultralytics YOLO**: Object detection for signatures and stamps
- **EasyOCR**: Optical Character Recognition for text extraction from signatures
- **Supervision**: Detection utilities and visualization

## Prerequisites

1. **Python 3.8+**
2. **Poppler**: Required for PDF to image conversion
   - Download from: https://github.com/oschwartz10612/poppler-windows/releases
   - Extract and note the path to the `bin` folder
   - Update the `POPPLER_PATH` in the scripts if needed

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download/Verify Models**:
   - The signature detection model will be automatically downloaded from HuggingFace on first run
   - Ensure `best.pt` (stamp detection model) is present in the project root

4. **Configure Poppler Path** (if needed):
   - Update `POPPLER_PATH` in:
     - `process_all_models.py` (line 18)
     - `app.py` (line 50)
     - `qr.py`, `signature.py`, `stamp.py` (if using individually)

## Project Structure

```
armeta/
├── app.py                      # Streamlit web application
├── process_all_models.py       # Main batch processing script
├── qr.py                       # QR code detection script
├── signature.py                # Signature detection script
├── stamp.py                    # Stamp detection script
├── main.py                     # (Contains commented code)
├── best.pt                     # Custom YOLO model for stamp detection
├── requirements.txt            # Python dependencies
├── model_outputs.json          # Generated JSON with all annotations
├── pdfs/                       # Directory containing PDF files
├── test/                       # Directory for PDFs to process
├── annotated_pages/            # Generated annotated images
├── qr1_detected_pages/         # QR detection outputs
├── sign_detected_pages/         # Signature detection outputs
└── stamp_detected_pages/       # Stamp detection outputs
```

## Usage

### Method 1: Batch Processing (Recommended)

Process all PDFs in the `test/` directory:

```bash
python process_all_models.py
```

This will:
- Process all PDFs in the `test/` directory
- Detect QR codes, signatures, and stamps on each page
- Generate `model_outputs.json` with all annotations
- Save annotated images to `annotated_pages/` directory

### Method 2: Web Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The web application provides:
- **Upload PDF**: Upload and process individual PDFs
- **Overview**: Dashboard with statistics and charts
- **PDF Explorer**: Browse processed PDFs and view annotations
- **All Annotations**: Searchable table of all annotations
- **Statistics**: Detailed statistical analysis
- **Image Viewer**: View annotated pages

**Note**: For the web app to display existing data, first run `process_all_models.py` to generate `model_outputs.json`.

### Method 3: Individual Scripts

You can also run individual detection scripts:

**QR Code Detection**:
```bash
python qr.py
```
(Update `pdf_path` in the script)

**Signature Detection**:
```bash
python signature.py
```
(Update `pdf_path` in the script)

**Stamp Detection**:
```bash
python stamp.py
```
(Update `pdf_path` in the script)

## Configuration

### Processing Settings

In `process_all_models.py`, you can adjust:

- `PDFS_DIR`: Directory containing PDFs to process (default: `"test"`)
- `OUTPUT_JSON`: Output JSON filename (default: `"model_outputs.json"`)
- `OUTPUT_IMAGES_DIR`: Directory for annotated images (default: `"annotated_pages"`)
- `POPPLER_PATH`: Path to Poppler bin directory

### Model Parameters

- **QR Detection**: `min_confidence=0.3` (in `process_all_models.py`)
- **Signature Detection**: `confidence > 0.1` threshold
- **Stamp Detection**: `conf=0.4` threshold

## Output Format

The system generates a JSON file with the following structure:

```json
{
  "document_name.pdf": {
    "page_1": {
      "annotations": [
        {
          "annotation_1": {
            "category": "qr",
            "bbox": {
              "x": 100.5,
              "y": 200.3,
              "width": 150.0,
              "height": 150.0
            },
            "area": 22500.0
          }
        }
      ],
      "page_size": {
        "width": 2550,
        "height": 3300
      }
    }
  }
}
```

### Visual Output

The system generates annotated images saved in the `annotated_pages/` directory. Each annotated image shows:
- **Bounding boxes** drawn around detected elements
- **Color-coded labels** (QR, SIGNATURE, STAMP) above each detection
- **Original page content** preserved with annotations overlaid

Example annotated images are available in:
- `annotated_pages/дозиметрия-2/page_4.png` - Contains signature and stamp
- `annotated_pages/лицензия-/page_1.png` - Contains multiple QR codes
- `annotated_pages/АПЗ-41-чб/` - Multiple pages with various annotations


