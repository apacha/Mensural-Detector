import argparse
import datetime
import os
from time import time

import keras
import numpy
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from reporting import sklearn_reporting
from reporting.TrainingHistoryPlotter import TrainingHistoryPlotter

from ClassWeightCalculator import ClassWeightCalculator
from models.ConfigurationFactory import ConfigurationFactory


def train_model(dataset_directory: str, model_name: str,
                width: int, height: int):
    print("Loading configuration and data-readers...")
    start_time = time()

    number_of_classes = len(os.listdir(os.path.join(dataset_directory, "training")))
    training_configuration = ConfigurationFactory.get_configuration_by_name(model_name, width, height,
                                                                            number_of_classes)

    train_generator = ImageDataGenerator(rotation_range=training_configuration.rotation_range,
                                         zoom_range=training_configuration.zoom_range)
    training_data_generator = DirectoryIterator(
        directory=os.path.join(dataset_directory, "training"),
        image_data_generator=train_generator,
        target_size=(training_configuration.input_image_rows,
                     training_configuration.input_image_columns),
        batch_size=training_configuration.training_minibatch_size,
    )
    training_steps_per_epoch = np.math.ceil(training_data_generator.samples / training_data_generator.batch_size)

    validation_generator = ImageDataGenerator()
    validation_data_generator = DirectoryIterator(
        directory=os.path.join(dataset_directory, "validation"),
        image_data_generator=validation_generator,
        target_size=(
            training_configuration.input_image_rows,
            training_configuration.input_image_columns),
        batch_size=training_configuration.training_minibatch_size)
    validation_steps_per_epoch = np.math.ceil(validation_data_generator.samples / validation_data_generator.batch_size)

    test_generator = ImageDataGenerator()
    test_data_generator = DirectoryIterator(
        directory=os.path.join(dataset_directory, "test"),
        image_data_generator=test_generator,
        target_size=(training_configuration.input_image_rows,
                     training_configuration.input_image_columns),
        batch_size=training_configuration.training_minibatch_size,
        shuffle=False)
    test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

    model = training_configuration.classifier()
    model.summary()

    print("Model {0} loaded.".format(training_configuration.name()))
    print(training_configuration.summary())

    start_of_training = datetime.date.today()
    best_model_path = "{0}_{1}.h5".format(start_of_training, training_configuration.name())

    monitor_variable = 'val_acc'

    model_checkpoint = ModelCheckpoint(best_model_path, monitor=monitor_variable, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor=monitor_variable,
                               patience=training_configuration.number_of_epochs_before_early_stopping,
                               verbose=1)
    learning_rate_reduction = ReduceLROnPlateau(monitor=monitor_variable,
                                                patience=training_configuration.number_of_epochs_before_reducing_learning_rate,
                                                verbose=1,
                                                factor=training_configuration.learning_rate_reduction_factor,
                                                min_lr=training_configuration.minimum_learning_rate)
    tensorboard_callback = TensorBoard(
        log_dir="./logs/{0}_{1}/".format(start_of_training, training_configuration.name()),
        batch_size=training_configuration.training_minibatch_size)

    callbacks = [model_checkpoint, early_stop, tensorboard_callback, learning_rate_reduction]

    class_weight_calculator = ClassWeightCalculator()
    balancing_method = "simple"
    class_weights = class_weight_calculator.calculate_class_weights(dataset_directory,
                                                                    method=balancing_method,
                                                                    class_indices=training_data_generator.class_indices)
    if balancing_method is not None:
        print("Using {0} method for obtaining class weights to compensate for an unbalanced dataset.".format(
            balancing_method))

    print("Training on dataset...")
    history = model.fit_generator(
        generator=training_data_generator,
        steps_per_epoch=training_steps_per_epoch,
        epochs=training_configuration.number_of_epochs,
        callbacks=callbacks,
        validation_data=validation_data_generator,
        validation_steps=validation_steps_per_epoch,
        class_weight=class_weights
    )

    print("Loading best model from check-point and testing...")
    best_model = keras.models.load_model(best_model_path)

    test_data_generator.reset()
    file_names = test_data_generator.filenames
    class_labels = os.listdir(os.path.join(dataset_directory, "test"))
    # Notice that some classes have so few elements, that they are not present in the test-set and do not
    # appear in the final report. To obtain the correct classes, we have to enumerate all non-empty class
    # directories inside the test-folder and use them as labels
    names_of_classes_with_test_data = [
        class_name for class_name in class_labels
        if os.listdir(os.path.join(dataset_directory, "test", class_name))]
    true_classes = test_data_generator.classes
    predictions = best_model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
    predicted_classes = numpy.argmax(predictions, axis=1)

    test_data_generator.reset()
    evaluation = best_model.evaluate_generator(test_data_generator, steps=test_steps_per_epoch)
    classification_accuracy = 0

    print("Reporting classification statistics with micro average")
    report = sklearn_reporting.classification_report(true_classes, predicted_classes, digits=3,
                                                     target_names=names_of_classes_with_test_data, average='micro')
    print(report)

    print("Reporting classification statistics with macro average")
    report = sklearn_reporting.classification_report(true_classes, predicted_classes, digits=3,
                                                     target_names=names_of_classes_with_test_data, average='macro')
    print(report)

    print("Reporting classification statistics with weighted average")
    report = sklearn_reporting.classification_report(true_classes, predicted_classes, digits=3,
                                                     target_names=names_of_classes_with_test_data, average='weighted'
                                                     )
    print(report)

    indices_of_misclassified_files = [i for i, e in enumerate(true_classes - predicted_classes) if e != 0]
    misclassified_files = [file_names[i] for i in indices_of_misclassified_files]
    misclassified_files_actual_prediction_indices = [predicted_classes[i] for i in indices_of_misclassified_files]
    misclassified_files_actual_prediction_classes = [class_labels[i] for i in
                                                     misclassified_files_actual_prediction_indices]
    print("Misclassified files:")
    for i in range(len(misclassified_files)):
        print("\t{0} is incorrectly classified as {1}".format(misclassified_files[i],
                                                              misclassified_files_actual_prediction_classes[i]))

    for i in range(len(best_model.metrics_names)):
        current_metric = best_model.metrics_names[i]
        print("{0}: {1:.5f}".format(current_metric, evaluation[i]))
        if current_metric == 'acc' or current_metric == 'output_class_acc':
            classification_accuracy = evaluation[i]
    print("Total Accuracy: {0:0.5f}%".format(classification_accuracy * 100))
    print("Total Error: {0:0.5f}%".format((1 - classification_accuracy) * 100))

    end_time = time()
    print("Execution time: {0:.1f}s".format(end_time - start_time))
    training_result_image = "{1}_{0}_{2:.1f}p.png".format(training_configuration.name(), start_of_training,
                                                          classification_accuracy * 100)
    TrainingHistoryPlotter.plot_history(history, training_result_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, that is used for storing the images during training")
    parser.add_argument("--model_name", type=str, default="vgg4",
                        help="The model used for training the network. Run ListAvailableConfigurations.ps1 or "
                             "models/ConfigurationFactory.py to get a list of all available configurations")
    parser.add_argument("--width", default=128, type=int, help="Width of the input-images for the network in pixel")
    parser.add_argument("--height", default=448, type=int,
                        help="Height of the input-images for the network in pixel")

    flags, unparsed = parser.parse_known_args()

    train_model(dataset_directory=flags.dataset_directory,
                model_name=flags.model_name,
                width=flags.width,
                height=flags.height)
