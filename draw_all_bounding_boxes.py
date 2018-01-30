import argparse

import os

from tqdm import tqdm

from draw_bounding_boxes import draw_bounding_boxes_into_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draws the bounding boxes for all image')
    parser.add_argument('-dataset_directory', dest='dataset_directory', type=str, default="db",
                        help='Path to the folder containing all images and annotations')
    args = parser.parse_args()

    dataset_directory = args.dataset_directory
    files = os.listdir(dataset_directory)
    images = sorted([file for file in files if file.endswith(".JPG")])
    annotations = sorted([file for file in files if file.endswith(".txt")])
    pairs = zip(images, annotations)

    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)

    for pair in tqdm(pairs, desc="Generating annotated images", total=len(images)):
        image = os.path.join(dataset_directory, pair[0])
        annotations = os.path.join(dataset_directory, pair[1])
        destination_file = os.path.join(output_directory,
                                        os.path.splitext(os.path.basename(image))[0] + "_annotated.JPG")
        draw_bounding_boxes_into_image(image, annotations, destination_file)
