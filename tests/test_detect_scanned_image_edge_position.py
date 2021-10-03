from src.detect_scanned_image_edge_position import detect_scanned_image_edge_position
from pathlib import Path
import cv2
import numpy as np
import pickle
import argparse


def test_detect_scanned_image_edge_position():
    # set arguments
    base_dir_path = Path.cwd()
    in_file_path = base_dir_path / "data/sample.jpg"
    exp_file_path = base_dir_path / "tests/exp/detect_scanned_image_edge_position.pkl"
    args = {"background_is_black": False}
    parser = argparse.ArgumentParser()
    parser.add_argument("--background_is_black", type=bool)
    args = parser.parse_args(["--background_is_black", False])

    # run test
    in_image = cv2.imread(str(in_file_path))
    result_image, result_data = detect_scanned_image_edge_position(in_image, args)

    # check result
    with open(exp_file_path, "rb") as f:
        exp_data = pickle.load(f)

    assert np.array_equal(result_data, exp_data)
