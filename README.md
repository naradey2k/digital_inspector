# Document Annotation System

A machine learning-powered document analysis system that automatically detects and annotates QR codes, signatures, and stamps in PDF documents. The system uses multiple deep learning models to identify these elements and provides a web-based interface for visualization and analysis.

## Features

- **QR Code Detection**: Detects QR codes in PDF documents using QReader
- **Signature Detection**: Identifies signatures using YOLOv8 model with OCR text extraction
- **Stamp Detection**: Detects stamps using a custom-trained YOLO model
- **Web Visualization**: Interactive Streamlit dashboard for viewing and analyzing results
- **Batch Processing**: Process multiple PDFs at once
- **Annotation Export**: Export results as JSON and CSV files

## Technologies Used

### Machine Learning & Computer Vision
- **QReader**: QR code detection and decoding
- **Ultralytics YOLO**: Object detection for signatures and stamps
- **EasyOCR**: Optical Character Recognition for text extraction from signatures
- **Supervision**: Detection utilities and visualization

### Web Framework & Visualization
- **Streamlit**: Web application framework for the interactive dashboard
- **Plotly**: Interactive charts and graphs
- **Pandas**: Data manipulation and analysis

### Image Processing
- **OpenCV**: Image processing and annotation drawing
- **Pillow (PIL)**: Image manipulation
- **pdf2image**: PDF to image conversion (requires Poppler)

### Other Libraries
- **NumPy**: Numerical operations
- **HuggingFace Hub**: Model downloading

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

## Annotation Examples

### QR Code Detection Example

```json
{
  "annotation_1": {
    "category": "qr",
    "bbox": {
      "x": 3499.875244140625,
      "y": 5752.39892578125,
      "width": 830.190673828125,
      "height": 830.5
    },
    "area": 689473.3546142578
  }
}
```

**Characteristics:**
- Detected at coordinates (x, y) with width and height
- Area calculated in pixels²
- Typically square-shaped with similar width and height

### Signature Detection Example

```json
{
  "annotation_2": {
    "category": "signature",
    "bbox": {
      "x": 1998.491455078125,
      "y": 4108.68896484375,
      "width": 576.539306640625,
      "height": 235.26806640625
    },
    "area": 135641.2878805399,
    "extracted_text": "Олегов @онишEка6 (Подпись) ЖАЙоKЕ₽Ujin 1 &orPix АСхАxх"
  }
}
```

**Characteristics:**
- Includes bounding box coordinates and dimensions
- **extracted_text**: OCR-extracted text from the signature region (expanded area)
- Typically wider than tall (rectangular shape)
- Text extraction includes surrounding text in the expanded region below the signature

### Stamp Detection Example

```json
{
  "annotation_3": {
    "category": "stamp",
    "bbox": {
      "x": 1151.23095703125,
      "y": 2391.40576171875,
      "width": 1006.68603515625,
      "height": 946.0947265625
    },
    "area": 952420.3491654396
  }
}
```

**Characteristics:**
- Detected with bounding box coordinates
- Typically larger area than signatures
- Often square or circular in shape
- No text extraction (stamps are visual elements)

### Complete Page Example

A page with multiple annotation types:

```json
{
  "page_4": {
    "annotations": [
      {
        "annotation_3": {
          "category": "signature",
          "bbox": {
            "x": 1998.491455078125,
            "y": 4108.68896484375,
            "width": 576.539306640625,
            "height": 235.26806640625
          },
          "area": 135641.2878805399,
          "extracted_text": "Олегов @онишEка6 (Подпись) ЖАЙоKЕ₽Ujin 1 &orPix АСхАxх"
        }
      },
      {
        "annotation_4": {
          "category": "stamp",
          "bbox": {
            "x": 1085.9072265625,
            "y": 4079.26318359375,
            "width": 985.36962890625,
            "height": 934.5673828125
          },
          "area": 920894.3151898384
        }
      }
    ],
    "page_size": {
      "width": 4959,
      "height": 7017
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

## Color Coding

Annotations are color-coded in the output images:
- **Red**: QR codes
- **Green**: Signatures
- **Blue**: Stamps

## Troubleshooting

### Poppler Not Found
- Ensure Poppler is installed and the path is correct
- Update `POPPLER_PATH` in all scripts

### Model Download Issues
- The signature model downloads automatically from HuggingFace
- Ensure you have internet connection on first run
- Check `huggingface_hub` is installed

### Memory Issues
- Process PDFs in smaller batches
- Reduce DPI in `convert_from_path` (currently 300)

### GPU Support
- EasyOCR can use GPU if available (set `gpu=True` in `easyocr.Reader`)
- YOLO models will use GPU automatically if CUDA is available

## License

This project uses various open-source libraries. Please refer to their respective licenses.

## Notes

- Processing time depends on PDF size and number of pages
- First run may take longer due to model downloads
- Ensure sufficient disk space for annotated images
- The system processes PDFs at 300 DPI for optimal detection accuracy

