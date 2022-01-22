import os
import tensorflow as tf
from datetime import datetime
from model.dataset import Dataset
from tensorflow.keras import layers
from util import logger
from util.config import config


class Trainer:
    def __init__(self):
        # image size
        self._size = config.model.size

        # input shape of images
        self._input_shape = (self._size, self._size, 3)

        # maximum epochs for early stopping
        self._epochs = config.model.max_epoch

        ds = Dataset()

        # split dataset to train and validation
        self._train_data, self._validation_data = ds.load()

        # number of classes
        self._num_classes = 3

    def _augmentation(self):
        """
        Data augmentation

        :return: Augmentation layers
        :rtype: tf.keras.Sequential
        """
        return tf.keras.Sequential([
            layers.RandomFlip('horizontal', input_shape=self._input_shape),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ])

    def _build(self):
        """
        Build model structure

        :return: CNN model
        :rtype: tf.keras.Sequential
        """
        model = tf.keras.Sequential([
            self._augmentation(),
            layers.Rescaling(1./255, input_shape=self._input_shape),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self._num_classes)
        ])

        # loss function for multi-classification
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # adam optimizer with custom learning rate
        optimizer = tf.keras.optimizers.Adam(0.00001)

        # compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # model summary
        logger.info('Model summary,')
        model.summary()

        return model

    def fit_model(self):
        """
        Fit model
        """
        # if you want to continue training
        if os.path.exists(config.model.checkpoint):
            model = tf.keras.models.load_model(config.model.checkpoint)
        else:
            model = self._build()

        time = datetime.now().strftime('%Y%m%d-%H%M%S')

        # early stopping callback
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        # log directory for tensorboard
        log_dir = f'{config.model.logdir}/{time}'

        # tensorboard to track the history of the model
        tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        # file path for checkpoint
        checkpoint = config.model.checkpoint

        # checkpoint to save model
        mc = tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor='loss',
            mode='min',
            save_best_only=True
        )

        # fit model
        model.fit(
            self._train_data,
            validation_data=self._validation_data,
            epochs=self._epochs,
            callbacks=[es, tb, mc]
        )
