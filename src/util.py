import cv2
from pathlib import Path
import argparse
import shutil


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
    colmn_num = 0
    csv_file_path = out_dir_path / out_csv_file_name
    if csv_file_path.exists():
        csv_file_path.unlink(missing_ok=False)

    for i, in_file_path in enumerate(in_dir_path.glob(f"*{in_suffix}")):
        # read
        image = cv2.imread(str(in_file_path))

        # process
        result_images, result_data = process_function(image, args)

        # save image
        if result_images is not None:
            for j, result_image in enumerate(result_images):
                out_file_path = out_dir_path / (
                    in_file_path.stem + f"_{j}" + out_suffix
                )
                cv2.imwrite(str(out_file_path), result_image)

        print(f"[{in_file_path.name}] process done")

        # save data(csv)
        with open(csv_file_path, mode="at") as f:
            if result_data is None:
                result_images, result_data = process_function(image, args)
                f.write(f"{in_file_path.name},")
                f.write(",".join(["NaN" for _ in range(colmn_num)]) + "\n")
            else:
                if i == 0:
                    # create csv and save header
                    f.write("file_name,")
                    f.write(",".join(list(result_data.keys())) + "\n")
                    colmn_num = len(list(result_data.keys()))
                f.write(f"{in_file_path.name},")
                f.write(",".join(list(result_data.values())) + "\n")
