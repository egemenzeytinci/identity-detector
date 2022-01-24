import gdown
from util.config import config
from zipfile import ZipFile


if __name__ == '__main__':
    # download from google drive
    gdown.download(config.loader.url, 'train.zip', quiet=False)

    # extract zip file
    with ZipFile('train.zip', 'r') as f:
        f.extractall()
