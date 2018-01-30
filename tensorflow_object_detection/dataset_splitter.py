import argparse
import os
import random
import shutil
from glob import glob
from typing import List
import re

import numpy
import numpy as np

from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """ Class that can be used to create a reproducible random-split of a dataset into train/validation/test sets """

    def __init__(self,
                 source_directory: str,
                 destination_directory: str):
        """
        :param source_directory: The root directory, where all images currently reside.
        :param destination_directory: The root directory, into which the data will be placed.
            Inside of this directory, the following structure will be created:

         destination_directory
         |- training
         |
         |- validation
         |
         |- test

        """
        self.source_directory = source_directory
        self.destination_directory = os.path.abspath(destination_directory)

    def get_random_training_validation_and_test_sample_indices(self,
                                                               dataset_size: int,
                                                               validation_percentage: float = 0.2,
                                                               test_percentage: float = 0.0,
                                                               seed: int = 0) -> (List[int], List[int], List[int]):
        """
        Returns a reproducible set of random sample indices from the entire dataset population
        :param dataset_size: The population size
        :param validation_percentage: the percentage of the entire population size that should be used for validation
        :param test_percentage: the percentage of the entire population size that should be used for testing
        :param seed: An arbitrary seed that can be used to obtain repeatable pseudo-random indices
        :return: A triple of three list, containing indices of the training, validation and test sets
        """
        random.seed(seed)
        all_indices = range(0, dataset_size)
        validation_sample_size = int(dataset_size * validation_percentage)
        test_sample_size = int(dataset_size * test_percentage)
        validation_sample_indices = random.sample(all_indices, validation_sample_size)
        test_sample_indices = random.sample((set(all_indices) - set(validation_sample_indices)), test_sample_size)
        training_sample_indices = list(set(all_indices) - set(validation_sample_indices) - set(test_sample_indices))
        return training_sample_indices, validation_sample_indices, test_sample_indices

    def delete_split_directories(self):
        print("Deleting split directories... ")
        shutil.rmtree(os.path.join(self.destination_directory, "train"), True)
        shutil.rmtree(os.path.join(self.destination_directory, "validation"), True)
        shutil.rmtree(os.path.join(self.destination_directory, "test"), True)

    def split_images_into_training_validation_and_test_set(self):
        print("Splitting data into training, validation and test sets...")

        number_of_images = len(glob(os.path.join(self.source_directory, "*.JPG")))
        training_sample_indices, validation_sample_indices, test_sample_indices = \
            self.get_random_training_validation_and_test_sample_indices(number_of_images)

        self.copy_files(self.source_directory, training_sample_indices, "training")
        self.copy_files(self.source_directory, validation_sample_indices, "validation")
        self.copy_files(self.source_directory, test_sample_indices, "test")

    def copy_files(self, path_to_images_of_class, sample_indices, name_of_split):
        files = numpy.array(glob(os.path.join(path_to_images_of_class, "*.JPG")))[sample_indices]
        destination_path = os.path.join(self.destination_directory, name_of_split)
        os.makedirs(destination_path, exist_ok=True)
        print("Copying {0} {1} files...".format(len(files), name_of_split))

        with open(os.path.join(self.destination_directory, name_of_split + ".txt"), "w") as image_set_dump:
            for image in files:
                image_set_dump.write(os.path.splitext(os.path.basename(image))[0] + "\n")
                shutil.copy(os.path.join(path_to_images_of_class, image), destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_directory",
        type=str,
        default="../db",
        help="The directory, where the images should be copied from")
    parser.add_argument(
        "--destination_directory",
        type=str,
        default="../training_validation_test",
        help="The directory, where the images should be split into the three directories 'train', 'test' and 'validation'")

    flags, unparsed = parser.parse_known_args()

    datasest = DatasetSplitter(flags.source_directory, flags.destination_directory)
    datasest.delete_split_directories()
    datasest.split_images_into_training_validation_and_test_set()
