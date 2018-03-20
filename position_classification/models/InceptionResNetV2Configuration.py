from keras import Input
from keras.engine import Layer, Model
from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Flatten, MaxPooling2D, AveragePooling2D, \
    add
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import plot_model
from keras.applications import InceptionResNetV2

from position_classification.models.TrainingConfiguration import TrainingConfiguration


class InceptionResNetV2Configuration(TrainingConfiguration):
    """ A network with residual modules """

    def __init__(self, width: int, height: int, number_of_classes: int):
        super().__init__(data_shape=(height, width, 3), number_of_classes=number_of_classes)

    def classifier(self) -> Model:
        """ Returns the model of this configuration """
        model = InceptionResNetV2(include_top=True, weights=None, input_shape=self.data_shape, pooling='avg', classes=self.number_of_classes)

        model.compile(self.get_optimizer(), loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def name(self) -> str:
        """ Returns the name of this configuration """
        return "inception_resnet_v2"


if __name__ == "__main__":
    configuration = InceptionResNetV2Configuration(448, 160, 32)
    classifier = configuration.classifier()
    classifier.summary()
    plot_model(classifier, to_file="inception_resnet_v2.png")
    print(configuration.summary())
