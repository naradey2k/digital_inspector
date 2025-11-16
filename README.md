# Document Annotation System

## Features

- **QR Code Detection**: Detects QR codes in PDF documents using QReader
- **Signature Detection**: Identifies signatures using YOLOv8 model with OCR text extraction
- **Stamp Detection**: Detects stamps using a custom-trained YOLO model
- **Web Visualization**: Interactive Streamlit dashboard for viewing and analyzing results

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

## 🧩 Usage

### Training Pipeline

Using `stamp_segmentation_kaggle.ipynb` and `stamp_detection_finetune.ipynb` use could see the training pipeline for fine-tuned YOLOv8 for stamp detection. 
All the data are in [Stamp Verification Dataset](https://www.kaggle.com/datasets/rtatman/stamp-verification-staver-dataset/data)

### Method 1: Batch Processing

Process all PDFs in the `test/` directory:

```bash
python process_all_models.py
```

This will:
- Process all PDFs in the `test/` directory
- Detect QR codes, signatures, and stamps on each page
- Generate `outputs.json` with all annotations
- Save annotated images to `annotated_pages/` directory

### Method 2: Web Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The web application provides:
- **Upload PDF**: Upload and process individual PDFs
- **Image Viewer**: View annotated pages

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

## 🖼️ Detection Results
<div style="text-align:center;">
  <figure style="display:inline-block; margin: 10px;">
  <img src="page_2.png" width="250"/>
  </figure>

  <figure style="display:inline-block; margin: 10px;">
  <img src="page_4.png" width="250"/>
  </figure>
</div>

