import tensorflow as tf
from tensorflow import keras
from keras_tuner import HyperModel

class WineQualityHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        """Builds a tunable neural network model."""
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=self.input_shape))

        # Tune the number of hidden layers
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(keras.layers.Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                activation=hp.Choice(f'activation_{i}', ['relu', 'tanh'])
            ))
            model.add(keras.layers.Dropout(
                rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
            ))

        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))

        # Tune the learning rate
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
