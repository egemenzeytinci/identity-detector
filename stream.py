import argparse
import cv2
from model.prediction import Predictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help='The best model path')

    # parse arguments
    args = parser.parse_args()
    cap = cv2.VideoCapture(0)

    predictor = Predictor(model_path=args.model)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # predict image class
        img_cls, prob = predictor.predict_class(frame, processed=False)

        # edit image class name
        title = img_cls.replace('_', ' ').title()

        # put text to stream
        cv2.putText(
            img=frame,
            text=f'{title} ({round(prob, 2)}%)',
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_4
        )

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
