import cv2
import numpy as np
import pytest

from face_detection import annotate_frame, detect_faces, load_detector


def test_load_detector_returns_loaded_cascade() -> None:
    assert not load_detector().empty()


def test_detect_faces_rejects_invalid_settings() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = load_detector()

    with pytest.raises(ValueError, match="scale_factor"):
        detect_faces(frame, detector, scale_factor=1)
    with pytest.raises(ValueError, match="min_neighbors"):
        detect_faces(frame, detector, min_neighbors=0)


def test_blank_frame_has_no_faces() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    assert detect_faces(frame, load_detector()) == ()


def test_annotate_frame_draws_count_label() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = annotate_frame(frame, ((10, 10, 20, 20),))

    assert result is frame
    assert int(result.sum()) > 0
