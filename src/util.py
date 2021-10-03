import cv2
from pathlib import Path
import argparse


def process_images_and_save(
    in_dir_path: Path,
    in_suffix: str,
    process_function: any,
    args: argparse.Namespace,
    out_dir_path: Path,
    out_suffix: str,
    out_csv_file_name: str,
):

    # create output folder
    if not out_dir_path.exists():
        out_dir_path.mkdir(parents=True)

    # "read -> process -> save" image files
    for i, in_file_path in enumerate(in_dir_path.glob(f"*{in_suffix}")):
        # read
        image = cv2.imread(str(in_file_path))

        # process
        result_image, result_data = process_function(image, args)

        # save image
        out_file_path = out_dir_path / (in_file_path.stem + out_suffix)
        cv2.imwrite(str(out_file_path), result_image)

        print(f"[{in_file_path.name}] process done" + " " * 30, end="")

        # save data(csv)
        csv_file_path = out_dir_path / out_csv_file_name
        with open(csv_file_path, mode="wt") as f:
            if i == 0:
                # create csv and save header
                f.write("file_name,")
                f.write(",".join(list(result_data.keys())) + "\n")
            f.write(f"{in_file_path.name},")
            f.write(",".join(list(result_data.values())) + "\n")
