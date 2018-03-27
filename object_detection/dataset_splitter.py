import argparse
import os
import random
import shutil
from glob import glob
from typing import List, Tuple
import re

import numpy


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
                                                               validation_percentage: float = 0.1,
                                                               test_percentage: float = 0.1,
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

    def get_random_cross_validation_sample_indices(self,
                                                   dataset_size: int,
                                                   number_of_splits=5,
                                                   seed: int = 0) -> (List[Tuple[List[int], List[int]]]):
        """
        Returns indices for train/test splits as used in n-fold cross validation
        :param dataset_size:
        :param number_of_splits:
        :param seed: An arbitrary seed that can be used to obtain repeatable pseudo-random indices
        :return: A list of pairs with train/test indices per split
        """
        random.seed(seed)
        all_indices = list(range(0, dataset_size))
        random.shuffle(all_indices)
        test_size = int(dataset_size / number_of_splits)
        splits = []
        for split in range(number_of_splits):
            first_index_of_batch = split * test_size
            last_index_of_batch = (split + 1) * test_size
            test_sample_indices = all_indices[first_index_of_batch: last_index_of_batch]
            training_sample_indices = list(set(all_indices) - set(test_sample_indices))
            splits.append((training_sample_indices, test_sample_indices))
        return splits

    def delete_split_directories(self) -> None:
        print("Deleting split directories... ")
        shutil.rmtree(os.path.join(self.destination_directory, "train"), True)
        shutil.rmtree(os.path.join(self.destination_directory, "validation"), True)
        shutil.rmtree(os.path.join(self.destination_directory, "test"), True)

    def split_images_into_training_validation_and_test_set(self, number_of_images: int) -> None:
        print("Splitting data into training, validation and test sets...")

        training_sample_indices, validation_sample_indices, test_sample_indices = \
            self.get_random_training_validation_and_test_sample_indices(number_of_images)

        self.copy_files(self.source_directory, self.destination_directory, training_sample_indices, "training")
        self.copy_files(self.source_directory, self.destination_directory, validation_sample_indices, "validation")
        self.copy_files(self.source_directory, self.destination_directory, test_sample_indices, "test")

    def copy_files(self, path_to_images_of_class, destination_directory, sample_indices, name_of_split) -> None:
        files = numpy.array(glob(os.path.join(path_to_images_of_class, "*.JPG")))[sample_indices]
        destination_path = os.path.join(destination_directory, name_of_split)
        os.makedirs(destination_path, exist_ok=True)
        print("Copying {0} {1} files...".format(len(files), name_of_split))

        with open(os.path.join(destination_directory, name_of_split + ".txt"), "w") as image_set_dump:
            for image in files:
                image_set_dump.write(os.path.splitext(os.path.basename(image))[0] + "\n")
                shutil.copy(os.path.join(path_to_images_of_class, image), destination_path)

    def split_images_into_train_test_for_cross_validation(self, dataset_size: int, number_of_splits: int) -> None:
        splits = self.get_random_cross_validation_sample_indices(dataset_size, number_of_splits)
        for index, (training_sample_indices, test_sample_indices) in enumerate(splits):
            split_directory = os.path.join(self.destination_directory, "split-{0}".format(index + 1))

            self.copy_files(self.source_directory, split_directory, training_sample_indices, "training")
            self.copy_files(self.source_directory, split_directory, test_sample_indices, "test")


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
    parser.add_argument(
        "--number_of_splits",
        type=int,
        default=None,
        required=False,
        help="If specified, the number of splits for cross-validation. If not specified, dataset will"
             "be split into training/validation/test sets.")

    flags, unparsed = parser.parse_known_args()

    source_directory = flags.source_directory
    destination_directory = flags.destination_directory
    number_of_splits = flags.number_of_splits

    dataset_splitter = DatasetSplitter(source_directory, destination_directory)
    dataset_splitter.delete_split_directories()

    number_of_images = len(glob(os.path.join(source_directory, "*.JPG")))

    if number_of_splits:
        dataset_splitter.split_images_into_train_test_for_cross_validation(number_of_images, number_of_splits)
    else:
        dataset_splitter.split_images_into_training_validation_and_test_set(number_of_images)
