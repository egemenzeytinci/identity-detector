import json
import numpy as np
import tensorflow as tf
from attrdict import AttrDict
from pathlib import Path


class Predictor:
    def __init__(self, model_path):
        # image size
        self._size = 150

        # laod best model
        self._model = tf.keras.models.load_model(model_path)

        self._extension = '.jpg'

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

    def _predict_class(self, img):
        """
        Predict class from image tensor or file path

        :param tf.python.framework.ops.EagerTensor img: image tensor
        :return: image class and probability score
        :rtype: tuple[str, float]
        """
        # predict image class
        predictions = self._model.predict(img)

        # probability scores
        score = tf.nn.softmax(predictions[0])

        # get class name by max probability score
        img_cls = self._class_names[np.argmax(score)]

        # probability score over a hundred
        prob = 100 * np.max(score)

        return img_cls, prob

    def predict(self, path):
        """
        Predict class from directory or file

        :param str path: file or directory path
        :return: list of predictions
        :rtype: list[dict]
        """
        path = Path(path)

        predictions = []

        # if the path is directory
        if path.is_dir():
            for k in path.rglob(f'*{self._extension}'):
                # read image from file path
                img = self._read(k)

                img_cls, prob = self._predict_class(img)

                d = AttrDict()

                # prediction result with file name
                d.name = k.name
                d.prediction = img_cls
                d.probability = prob

                predictions.append(dict(d))
        else:
            # read image from file path
            img = self._read(path)

            img_cls, prob = self._predict_class(img)

            d = AttrDict()

            # prediction result with file name
            d.name = path.name
            d.prediction = img_cls
            d.probability = prob

            predictions.append(dict(d))

        return json.dumps(predictions, indent=4)
