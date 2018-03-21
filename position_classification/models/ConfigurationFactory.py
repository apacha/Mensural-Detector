from typing import List

from position_classification.models.InceptionResNetV2Configuration import InceptionResNetV2Configuration
from position_classification.models.InceptionResNetV2PretrainedConfiguration import \
    InceptionResNetV2PretrainedConfiguration
from position_classification.models.ResNet4Configuration import ResNet4Configuration
from position_classification.models.TrainingConfiguration import TrainingConfiguration
from position_classification.models.Vgg4Configuration import Vgg4Configuration
from position_classification.models.VggGlobalAverageConfiguration import VggGlobalAverageConfiguration


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
                          VggGlobalAverageConfiguration(width, height, number_of_classes),
                          InceptionResNetV2Configuration(width, height, number_of_classes),
                          InceptionResNetV2PretrainedConfiguration(width, height, number_of_classes),
                          ResNet4Configuration(width, height, number_of_classes)]
        return configurations


if __name__ == "__main__":
    configurations = ConfigurationFactory.get_all_configurations(1, 1, 1)
    print("Available configurations are:")
    for configuration in configurations:
        print("- " + configuration.name())
