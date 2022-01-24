# Identity Card Detector
National ID cards detector

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

Then train your model as follows,

```bash
$ python3 train.py ./default.ini # or custom ini file
```

Please notice that, you need to change the path variables in the default config file named as `default.ini`.

You can follow your model on the tensorboard,

```bash
tensorboard --logdir /tmp/tensorboard-logs
```

If you change `logdir` variable under the model section in the default config, you need to change the path to `/tmp/tensorboard-logs`.

## Prediction

You can download model [here](https://drive.google.com/file/d/1K8A6og7Q4lI-UmH8h-YHZw0EJKTMDlow).

Then, predict with using folder or only one file as follows,

```bash
python3 predict.py --model /path/to/model --path /path/to/folder # or /path/to/file
```
