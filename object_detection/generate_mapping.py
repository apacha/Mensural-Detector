import os
import glob
from collections import Counter
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_dir", default="../db", type=str)
parser.add_argument("-m", "--mapping_filepath", default="mapping.txt", type=str)
args = parser.parse_args()

dataset_directory = args.dataset_dir
mapping_filepath = args.mapping_filepath

annotation_files = glob.glob(os.path.join(dataset_directory, "*.txt"))
classes = []
for annotation_file in tqdm(annotation_files):
    with open(os.path.join(dataset_directory, annotation_file), 'r') as file:
        lines = file.read().splitlines()

    for line in lines:
        upper_left, lower_right, class_name, gt_position = line.split(';')
        classes.append(class_name)

classes = sorted(list(set(classes)))

with open(mapping_filepath, "w") as f:
    for i, classname in enumerate(classes):
        f.write("""
item{{
  id: {}
  name: '{}'
}}
""".format(i + 1, classname))
