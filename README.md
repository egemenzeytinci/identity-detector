# Identity Card Detector

You can check the application from the website below,

https://identity-detector.herokuapp.com

## Install

Install dependencies by using `pip`,

```bash
$ pip3.9 install -r requirements.txt
```

## Training

You need to download dataset with using download script,

```bash
$ python3 download.py ./default.ini # or custom ini file
```

Then train your model as follows.

**Please keep in mind**, if you've a saved model in the checkpoint path, the command below continues the training.

```bash
$ python3 train.py ./default.ini # or custom ini file
```

**Please notice that**, you need to change the path variables in the default config file named as `default.ini`.

You can follow your model on the tensorboard,

```bash
$ tensorboard --logdir /tmp/tensorboard-logs
```

**Please notice that**, If you change `logdir` variable under the model section in the default config, you need to change the path to `/tmp/tensorboard-logs`.

## Prediction

You can download model [here](https://drive.google.com/file/d/1dUAbGskgIqBWs86ut3m9Azx93_QOyV0u).

Then, predict with using folder or only one file as follows,

```bash
$ python3 predict.py --model /path/to/model --path /path/to/folder # or /path/to/file
```

Here is the example response for a folder,

```json
[
    {
        "name": "identity_card.jpg",
        "prediction": "document",
        "probability": 99.99996423721313
    },
    {
        "name": "non_document.jpg",
        "prediction": "non_document",
        "probability": 100.0
    },
    {
        "name": "selfie.jpg",
        "prediction": "selfie",
        "probability": 99.23766255378723
    }
]
```

Or you can use your camera,

```bash
$ python3 stream.py --model /path/to/model
```

## Examples

![](image/example.png)
