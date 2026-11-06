# OpenCV Face Detection

A small Python project that detects faces with OpenCV's bundled Haar cascade.
It supports both a still image and a live webcam feed.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Detect faces in an image

The annotated image is written beside the input as
`photo_detected.jpg` unless `--output` is supplied:

```powershell
python face_detection.py --image photo.jpg
python face_detection.py --image photo.jpg --output results\photo.jpg
```

## Detect faces with a webcam

```powershell
python face_detection.py
python face_detection.py --camera 1
```

Press **Q** or **Esc** to close the webcam window.

## Adjust detection

Lower `--min-neighbors` to detect more possible faces, or increase it to
reduce false positives:

```powershell
python face_detection.py --image group.jpg --min-neighbors 4 --scale-factor 1.05
```

The detector is intended for frontal faces and works best with clear,
well-lit images. It is not a recognition system and does not identify people.

## Test

```powershell
pytest
```
