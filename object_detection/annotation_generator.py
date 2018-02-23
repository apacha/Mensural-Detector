import argparse
import os
from itertools import groupby

from PIL import Image
from lxml import etree
from typing import List, Tuple

from lxml.etree import Element, SubElement
import scipy.ndimage
from tqdm import tqdm


def create_annotations_in_pascal_voc_format(annotations_directory: str,
                                            file_name: str,
                                            objects_appearing_in_image: List[
                                                Tuple[str, str, Tuple[int, int, int, int]]],
                                            image_width: int,
                                            image_height: int,
                                            number_of_channels: int):
    os.makedirs(annotations_directory, exist_ok=True)

    annotation = Element("annotation")
    folder = SubElement(annotation, "folder")
    folder.text = "capitan_images"
    filename = SubElement(annotation, "filename")
    filename.text = file_name
    source = SubElement(annotation, "source")
    database = SubElement(source, "database")
    database.text = "CAPITAN"
    source_annotation = SubElement(source, "annotation")
    source_annotation.text = "CAPITAN (v0.1)"
    image = SubElement(source, "image")
    image.text = "CAPITAN"
    size = SubElement(annotation, "size")
    width = SubElement(size, "width")
    width.text = str(image_width)
    height = SubElement(size, "height")
    height.text = str(image_height)
    depth = SubElement(size, "depth")
    depth.text = str(number_of_channels)

    # Write results to file
    for detected_object in objects_appearing_in_image:
        class_name = detected_object[0]
        position = detected_object[1]
        bounding_box = detected_object[2]
        xmin, ymin, xmax, ymax = bounding_box

        object = SubElement(annotation, "object")
        name = SubElement(object, "name")
        name.text = class_name
        pose = SubElement(object, "pose")
        pose.text = position
        truncated = SubElement(object, "truncated")
        truncated.text = "0"
        difficult = SubElement(object, "difficult")
        difficult.text = "0"
        bb = SubElement(object, "bndbox")
        bb_xmin = SubElement(bb, "xmin")
        bb_xmin.text = str(xmin)
        bb_ymin = SubElement(bb, "ymin")
        bb_ymin.text = str(ymin)
        bb_xmax = SubElement(bb, "xmax")
        bb_xmax.text = str(xmax)
        bb_ymax = SubElement(bb, "ymax")
        bb_ymax.text = str(ymax)

    xml_file_path = os.path.join(annotations_directory, os.path.splitext(file_name)[0] + ".xml")
    pretty_xml_string = etree.tostring(annotation, pretty_print=True)

    with open(xml_file_path, "wb") as xml_file:
        xml_file.write(pretty_xml_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates Annotations in Pascal VOC format')
    parser.add_argument('-dataset_directory', dest='dataset_directory', type=str, default="../db",
                        help='Path to the folder containing all images and annotations')
    parser.add_argument('-annotations_directory', dest='annotations_directory', type=str, default="../annotations",
                        help='Path to the folder containing all images and annotations')
    args = parser.parse_args()

    dataset_directory = args.dataset_directory
    annotations_directory = args.annotations_directory
    files = os.listdir(dataset_directory)
    images = sorted([file for file in files if file.endswith(".JPG")])
    annotations = sorted([file for file in files if file.endswith(".txt")])
    pairs = zip(images, annotations)

    for file_name, annotation_path in tqdm(pairs, desc="Generating annotations", total=len(images)):
        with open(os.path.join(dataset_directory, annotation_path), 'r') as gt_file:
            lines = gt_file.read().splitlines()

        objects_appearing_in_image = []  # type: List[Tuple[str, str, Tuple[int, int, int, int]]]
        for line in lines:
            upper_left, lower_right, class_name, gt_position = line.split(';')
            xmin, ymin = upper_left.split(',')
            xmax, ymax = lower_right.split(',')
            objects_appearing_in_image.append((class_name, gt_position, (xmin, ymin, xmax, ymax)))

        height, width, channels = scipy.ndimage.imread(os.path.join(dataset_directory,
                                                                    file_name)).shape

        create_annotations_in_pascal_voc_format(annotations_directory, file_name,
                                                objects_appearing_in_image, width,
                                                height, channels)
