"""Face detection with OpenCV using an image file or a webcam."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Detect faces in an image or from a webcam using OpenCV."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image",
        type=Path,
        help="Path to an input image. If omitted, the webcam is used.",
    )
    source.add_argument(
        "--camera",
        type=int,
        metavar="INDEX",
        help="Webcam index to use (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save the annotated image to this path when --image is used.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Image scale between detection passes (default: 1.1).",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Required neighboring detections (default: 5).",
    )
    return parser


def load_detector() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled frontal-face Haar cascade."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Unable to load face detector: {cascade_path}")
    return detector


def detect_faces(
    frame: cv2.typing.MatLike,
    detector: cv2.CascadeClassifier,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> tuple[cv2.typing.Rect, ...]:
    """Return detected face rectangles for a BGR frame."""
    if scale_factor <= 1:
        raise ValueError("scale_factor must be greater than 1")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be at least 1")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30),
    )
    return tuple(faces)


def annotate_frame(
    frame: cv2.typing.MatLike,
    faces: Sequence[cv2.typing.Rect],
) -> cv2.typing.MatLike:
    """Draw face rectangles and a count label on a frame."""
    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def process_image(
    image_path: Path,
    output_path: Path | None,
    detector: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
) -> int:
    """Detect and annotate faces in one image."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    faces = detect_faces(frame, detector, scale_factor, min_neighbors)
    annotate_frame(frame, faces)
    destination = output_path or image_path.with_name(f"{image_path.stem}_detected{image_path.suffix}")
    if not cv2.imwrite(str(destination), frame):
        raise OSError(f"Unable to write output image: {destination}")
    print(f"Detected {len(faces)} face(s). Saved: {destination}")
    return 0


def run_camera(
    camera_index: int,
    detector: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
) -> int:
    """Run real-time detection until the user presses Q or Escape."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera {camera_index}")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Unable to read a frame from the camera")
            faces = detect_faces(frame, detector, scale_factor, min_neighbors)
            annotate_frame(frame, faces)
            cv2.imshow("OpenCV Face Detection (Q/Esc to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected face-detection mode."""
    args = build_parser().parse_args(argv)
    detector = load_detector()
    if args.image is not None:
        return process_image(
            args.image,
            args.output,
            detector,
            args.scale_factor,
            args.min_neighbors,
        )
    camera_index = 0 if args.camera is None else args.camera
    return run_camera(camera_index, detector, args.scale_factor, args.min_neighbors)


if __name__ == "__main__":
    raise SystemExit(main())
