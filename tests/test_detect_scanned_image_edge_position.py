from src.detect_scanned_image_edge_position import detect_scanned_image_edge_position
from pathlib import Path
import cv2
import numpy as np


def test_idetect_scanned_image_edge_position():
    # set arguments
    base_dir_path = Path.cwd()
    in_file_path = base_dir_path / "data/sample.jpg"
    exp_file_path = base_dir_path / "tests/exp/image_process_sample.png"

    # read images
    in_image = cv2.imread(str(in_file_path))
    exp_image = cv2.imread(str(exp_file_path))

    debug_image, result_data = detect_scanned_image_edge_position(in_image, None)

    assert np.array_equal(debug_image, exp_image)
