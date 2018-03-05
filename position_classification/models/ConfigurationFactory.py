from typing import List

from models.ResNet4Configuration import ResNet4Configuration
from models.TrainingConfiguration import TrainingConfiguration
from models.Vgg4Configuration import Vgg4Configuration


class ConfigurationFactory:
    @staticmethod
    def get_configuration_by_name(name: str,
                                  width: int,
                                  height: int,
                                  number_of_classes: int) -> TrainingConfiguration:

        configurations = ConfigurationFactory.get_all_configurations(width, height, number_of_classes)

        for i in range(len(configurations)):
            if configurations[i].name() == name:
                return configurations[i]

        raise Exception("No configuration found by name {0}".format(name))

    @staticmethod
    def get_all_configurations(width, height, number_of_classes) -> List[
        TrainingConfiguration]:
        configurations = [Vgg4Configuration(width, height, number_of_classes),
                          ResNet4Configuration(width, height, number_of_classes)]
        return configurations


if __name__ == "__main__":
    configurations = ConfigurationFactory.get_all_configurations(1, 1, 1)
    print("Available configurations are:")
    for configuration in configurations:
        print("- " + configuration.name())
