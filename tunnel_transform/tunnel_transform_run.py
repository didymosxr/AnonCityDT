import sys
import argparse
from pathlib import Path
from tunnel_transform import COCO_Transform_Tunnelvision


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--images", type=str, help="Path to image directory", required=True
    )
    parser.add_argument(
        "-j", "--json", type=str, help="Path to Coco json file", required=True
    )
    parser.add_argument(
        "-o", "--out", type=str, help="Path to Output directory", required=True
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        help="Scaling of the tunnel vision effect",
        required=True,
    )

    args = parser.parse_args()
    return args


def fisheye_transform(args):
    try:
        img_path = Path(args.images)
        anno_path = Path(args.json)

        if not img_path.exists():
            raise FileNotFoundError("Path to images does not exists!")
        elif not anno_path.exists():
            raise FileNotFoundError("Path to json file does not exists!")

        tunnel_transformer = COCO_Transform_Tunnelvision(
            image_path=args.images,
            annotation_path=args.json,
            output_path=args.out,
            strength=args.scale,
        )
        tunnel_transformer.transform_coco_and_images()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main(arglist):

    sys.argv = arglist
    args = parse_args()

    fisheye_transform(args)


if __name__ == "__main__":

    main(sys.argv)
