"""Command line interface module for OpenCV-based python environment sample repository.
"""

import argparse
from distutils.util import strtobool
from pathlib import Path
from util import process_images_and_save
from detect_scanned_image_edge_position import detect_scanned_image_edge_position


def parseArgs():
    parser = argparse.ArgumentParser(description="python image processing program")

    parser.add_argument(
        "--background_is_black",
        type=strtobool,
        default=True,
        help="Background Color(False:White, True:Black)",
    )
    parser.add_argument(
        "--in_dir_path",
        type=str,
        default="data/sample.jpg",
        help="input image directory path",
    )
    parser.add_argument(
        "--in_suffix", type=str, default=".png", help="input image file extension"
    )

    parser.add_argument(
        "--out_dir_path", type=str, default="out/", help="output image directory path"
    )

    parser.add_argument(
        "--out_suffix", type=str, default=".png", help="output image file extension"
    )

    parser.add_argument(
        "--out_csv_file_name",
        type=str,
        default="result.csv",
        help="output csv file name",
    )

    args = parser.parse_args()

    # convert string argument to Path
    args.in_dir_path = Path(args.in_dir_path)
    args.out_dir_path = Path(args.out_dir_path)

    return args


def main():
    args = parseArgs()

    process_function = detect_scanned_image_edge_position

    process_images_and_save(
        in_dir_path=args.in_dir_path,
        in_suffix=args.in_suffix,
        args=args,
        process_function=process_function,
        out_dir_path=args.out_dir_path,
        out_suffix=args.out_suffix,
        out_csv_file_name=args.out_csv_file_name,
    )


if __name__ == "__main__":
    main()
