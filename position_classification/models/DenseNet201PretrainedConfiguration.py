from keras.applications import DenseNet201
from keras.engine import Model
from keras.layers import Activation, Convolution2D, GlobalAveragePooling2D
from keras.utils import plot_model

from position_classification.models.TrainingConfiguration import TrainingConfiguration


class DenseNet201PretrainedConfiguration(TrainingConfiguration):
    """ A network with residual modules """

    def __init__(self, width: int, height: int, number_of_classes: int):
        super().__init__(data_shape=(height, width, 3), number_of_classes=number_of_classes)

    def classifier(self) -> Model:
        """ Returns the model of this configuration """
        base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=self.data_shape, pooling=None)
        x = base_model.output
        x = Convolution2D(self.number_of_classes, kernel_size=(1, 1), padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='output_class')(x)
        model = Model(inputs=base_model.inputs, outputs=x)
        model.compile(self.get_optimizer(), loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    def name(self) -> str:
        """ Returns the name of this configuration """
        return "dense_net_201"


if __name__ == "__main__":
    configuration = DenseNet201PretrainedConfiguration(448, 200, 32)
    classifier = configuration.classifier()
    classifier.summary()
    plot_model(classifier, to_file="dense_net_201.png")
    print(configuration.summary())
