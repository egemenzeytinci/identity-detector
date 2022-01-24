import gdown
import os
from util.config import config
from zipfile import ZipFile


if __name__ == '__main__':
    # if download folder does not exist
    if not os.path.exists(config.loader.path):
        os.makedirs(config.loader.path)

    destination = f'{config.loader.path}/train.zip'

    # download from google drive
    gdown.download(config.loader.url, destination, quiet=False)

    # extract zip file
    with ZipFile(destination, 'r') as f:
        f.extractall(config.loader.path)

    os.remove(destination)
