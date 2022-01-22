import numpy as np
import tensorflow as tf
from util.config import config


class Prediction:
    def __init__(self):
        # image size
        self._size = config.model.size

        # laod best model
        self._model = tf.keras.models.load_model(config.model.checkpoint)

        # define class names
        # notice that order is important
        self._class_names = [
            'document',
            'non_document',
            'selfie',
        ]

    def _read(self, path):
        """
        Read image file from path

        :param str path: image path
        :return: image eager tensor
        :rtype: tf.python.framework.ops.EagerTensor
        """
        # load image from path
        img = tf.keras.utils.load_img(
            path,
            target_size=(self._size, self._size)
        )

        # image to array
        img_array = tf.keras.utils.img_to_array(img)

        return tf.expand_dims(img_array, 0)

    def predict(self, img=None, path=None):
        """
        Predict class from image tensor or file path

        :param tf.python.framework.ops.EagerTensor img: image tensor
        :return: image class and probability score
        :rtype: tuple[str, float]
        """
        # path or image tensor directly
        if path:
            img = self._read(path)

        # predict image
        predictions = self._model.predict(img)

        # probability scores
        score = tf.nn.softmax(predictions[0])

        # get class name by max probability score
        img_class = self._class_names[np.argmax(score)]

        # probability score over a hundred
        max_score = 100 * np.max(score)

        return img_class, max_score
