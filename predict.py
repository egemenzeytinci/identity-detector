import argparse
from model.prediction import Predictor
from util import logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', help='File or folder path')
    parser.add_argument('--model', help='The best model path')

    # parse arguments
    args = parser.parse_args()

    predictor = Predictor(model_path=args.model)

    # train model according to remote content
    predictions = predictor.predict(args.path)

    logger.info(f'\n{predictions}')
