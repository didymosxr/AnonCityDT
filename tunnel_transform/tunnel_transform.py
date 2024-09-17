import json
import numpy as np
import os
import cv2
from tqdm import tqdm
from pathlib import Path
import shutil
import tempfile
from scipy.optimize import minimize_scalar
import copy


class COCO_Transform_Tunnelvision:
    def __init__(
        self, image_path: str, annotation_path: str, output_path: str, strength: float
    ):
        """This class takes a given COCO Dataset and
        creates a transformed version of it, applying a tunnel effect
        onto each image and corresponding data in json file.

        Args:
            image_path (str): Path to the image folder
            annotation_path (str): Path the to json file
            output_path (str): Path to the desired output folder
            strength (float): intensity of the tunnel effect
        """
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.output_path = output_path
        self.strength = strength

    def inverse_tunnel_transform_coordinates(self, new_x, new_y, cx, cy):
        try:
            # Calculate the transformed coordinates relative to the center
            nx = (new_x - cx) / cx
            ny = (new_y - cy) / cy

            # Calculate the radius and angle
            r_transformed = np.sqrt(nx**2 + ny**2)
            theta = np.arctan2(ny, nx)
            strength = self.strength

            def find_original_radius(r):
                """
                if r is negative, make it very large to exclude it from consideration
                in the optimization process.
                """
                if r < 0:
                    return np.inf
                spread_factor = 1 + (strength * (1 - (r ** (1 / 4))))
                return abs(r_transformed - (r * spread_factor))

            # Use a numerical method to find the root (original radius)
            result = minimize_scalar(
                find_original_radius, bracket=(0, r_transformed), method="brent"
            )
            # result = minimize_scalar(
            #     find_original_radius, bounds=(0, r_transformed), method="bounded"
            # )
            if not result.success:
                raise ValueError("Optimization did not converge.")

            original_radius = result.x
            # Calculate the original coordinates
            original_x = cx + original_radius * cx * np.cos(theta)
            original_y = cy + original_radius * cy * np.sin(theta)

            return original_x, original_y

        except ZeroDivisionError as e:
            print("Error: Division by zero encountered. Check the input values.", e)

        except ValueError as e:
            print("Value error encountered:", e)

        except Exception as e:
            print("An unexpected error occurred:", e)

    def tunnel_transform_coordinates(self, x, y, cx, cy):
        try:
            # Normalize coordinates to the center
            nx = (x - cx) / cx
            ny = (y - cy) / cy

            # Calculate the polar coordinates of (n_x, n_y)
            r = np.sqrt(nx**2 + ny**2)
            theta = np.arctan2(ny, nx)

            # Use a nonlinear transformation to spread the coordinates outward
            spread_factor = 1 + (self.strength * (1 - (r ** (1 / 4))))
            r_transformed = r * spread_factor
            # Calculate the Cartesian coordinates
            new_x = int(cx + r_transformed * cx * np.cos(theta))
            new_y = int(cy + r_transformed * cy * np.sin(theta))

            return new_x, new_y

        except Exception as e:
            print("An unexpected error occurred:", e)

    def tunnel_transform_image(self, image, h, w):
        transformed_img = np.zeros_like(image)
        cx, cy = w // 2, h // 2

        for y in range(h):
            for x in range(w):
                new_x, new_y = self.tunnel_transform_coordinates(x, y, cx, cy)
                if 0 <= new_x < w and 0 <= new_y < h:
                    transformed_img[y, x] = image[new_y, new_x]

        return transformed_img

    def transform_coco_and_images(self):
        """This method is responsible for the entire transformation process.
        The images are transformed and saved here and their annotations are adapted in
        a new json file.
        """
        # Load the COCO JSON file
        with open(self.annotation_path, "r") as f:
            coco_data = json.load(f)

        # Create output folders if they doesn't exist
        if Path(self.output_path).is_dir():
            shutil.rmtree(self.output_path)

        img_path = Path(self.output_path).joinpath("images")
        anno_path = Path(self.output_path).joinpath("annotations")
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(anno_path, exist_ok=True)

        cx, cy = None, None

        for annotation in tqdm(coco_data["annotations"], desc="Transformation Process"):
            # Get the image id and find the corresponding image
            image_id = annotation["image_id"]
            image_info = next(
                item for item in coco_data["images"] if item["id"] == image_id
            )
            image_filename = image_info["file_name"]
            image_path = os.path.join(self.image_path, image_filename)

            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue

            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2

            # Transform the image
            output_image_path = Path.joinpath(img_path, image_filename)

            if not output_image_path.is_file():
                transformed_img = self.tunnel_transform_image(img, h, w)
                cv2.imwrite(output_image_path, transformed_img)

            transformed_segmentation = []
            for polygon in annotation["segmentation"]:
                transformed_polygon = []
                for i in range(0, len(polygon), 2):
                    x, y = polygon[i], polygon[i + 1]
                    inverse_x, inverse_y = self.inverse_tunnel_transform_coordinates(
                        x, y, cx, cy
                    )
                    new_x, new_y = self.clip_coordinates(inverse_x, inverse_y, w, h)
                    transformed_polygon.append(new_x)
                    transformed_polygon.append(new_y)

                transformed_segmentation.append(transformed_polygon)

            transformed_bbox = self.transform_bbox(transformed_segmentation, w, h)

            annotation["segmentation"] = transformed_segmentation
            annotation["bbox"] = transformed_bbox

        # Save transformed COCO Json into a temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir).joinpath("temp_data.json")
            with open(temp_file_path, "w") as json_file:
                json.dump(coco_data, json_file)

            with open(temp_file_path, "r") as temp_json:
                temp_coco = json.load(temp_json)

            annotations = copy.deepcopy(temp_coco["annotations"])
            # pbar_clean_up = tqdm(total=len(annotations))
            for anno in tqdm(temp_coco["annotations"], desc="Clean up process:"):
                # pbar_clean_up.set_description("Clean up process:")
                if float(anno["bbox"][2]) <= 0.0 or float(anno["bbox"][3]) <= 0.0:
                    annotations.remove(anno)

            temp_coco["annotations"] = annotations

        # Save the transformed COCO JSON file
        output_json_path = Path.joinpath(anno_path, "coco_transformed.json")
        with open(output_json_path, "w") as f:
            json.dump(temp_coco, f)

        # Save transformed images without annotations
        self.transform_img_without_annotations()

    def transform_img_without_annotations(self):

        # Load annotation file
        with open(self.annotation_path, "r") as f:
            coco_data = json.load(f)

        img_path = Path(self.output_path).joinpath("images")

        # Get all the image id's from the annotations
        image_ids_anno = []
        for anno in coco_data["annotations"]:
            image_id = anno["image_id"]
            image_ids_anno.append(image_id)

        # Get all the image id's which have no annotations
        images_without_anno = []
        for img in coco_data["images"]:
            if img["id"] not in image_ids_anno:
                images_without_anno.append(img)

        # Transform all the images without annotation
        for img_coco in tqdm(
            images_without_anno, desc="Processing all images without annotations"
        ):
            image_filename = img_coco["file_name"]
            image_path = os.path.join(self.image_path, image_filename)
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue

            h, w = img.shape[:2]
            # Transform the image
            output_image_path = Path.joinpath(img_path, image_filename)

            if not output_image_path.is_file():
                transformed_img = self.tunnel_transform_image(img, h, w)
                cv2.imwrite(output_image_path, transformed_img)

    def clip_coordinates(self, x, y, width, height):
        """This function checks if the given coordinates (x, y)
        are in the boundary of an given image.

        Args:
            x (float or int): x coordinate of the point
            y (float or int): y coordinate of the point
            width (float or int): width of the image
            height (float or int): height of the image

        Returns:
            tuple: if one of the coordinates is outside of
            the image boundaries return 0
        """
        x_clipped = max(0, min(x, width - 1))
        y_clipped = max(0, min(y, height - 1))
        return x_clipped, y_clipped

    def transform_bbox(self, transformed_seg, width, height) -> list:
        """This method calculates the bounding box for the
        given transformed segmentation mask. The parameters
        width and height are used to check if the box is inside
        the image.

        Args:
            transformed_seg (list): The transformed segmentation mask
            width (int or float): width of the image
            height (int or float): height of the image

        Returns:
            list: Bounding Box in the format [x_min, y_min, w, h]
        """
        trans_x_coords = [
            new_x
            for polygon in transformed_seg
            for i in range(0, len(polygon), 2)
            for new_x in [polygon[i]]
        ]
        trans_y_coords = [
            new_y
            for polygon in transformed_seg
            for i in range(1, len(polygon), 2)
            for new_y in [polygon[i]]
        ]

        if trans_x_coords and trans_y_coords:
            trans_x_min = min(trans_x_coords)
            trans_y_min = min(trans_y_coords)
            trans_x_max = max(trans_x_coords)
            trans_y_max = max(trans_y_coords)

            x_min, y_min = self.clip_coordinates(
                trans_x_min, trans_y_min, width, height
            )
            x_max, y_max = self.clip_coordinates(
                trans_x_max, trans_y_max, width, height
            )

            transformed_bbox = [
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min,
            ]
            return transformed_bbox
