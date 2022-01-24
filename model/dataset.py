import tensorflow as tf
from util import logger
from util.config import config


class Dataset:
    def __init__(self):
        # path of the data
        self._path = f'{config.loader.path}/train'

        # image size
        self._size = config.loader.size

        # batch size
        self._batch_size = config.loader.batch_size

    def load(self):
        """
        Load dataset and split to train and validation

        return: train and validation dataset
        :rtype: tuple[BatchDataset, BatchDataset]
        """
        logger.info('Dataset items,')

        train = tf.keras.utils.image_dataset_from_directory(
            self._path,
            validation_split=0.2,
            subset='training',
            image_size=(self._size, self._size),
            batch_size=self._batch_size
        )

        validation = tf.keras.utils.image_dataset_from_directory(
            self._path,
            validation_split=0.2,
            subset='validation',
            image_size=(self._size, self._size),
            batch_size=self._batch_size
        )

        return train, validation
