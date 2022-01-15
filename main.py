import argparse
from src.KerasTrain import KerasTrain


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict classes of an image')
    parser.add_argument('-i', '--image', type=str,
                        help="Image path (default: None)", dest="img_path", default=None)
    parser.add_argument('-tr', '--train', type=bool,
                        help="Train the model", dest="train", default=False)
    parser.add_argument('-mp', '--model_path', type=str, help="Path to the model (default: model.h5)",
                        dest="model_path", default="model.h5")

    parser.add_argument('-e', '--epochs', type=int,
                        help="Epoch size (default: 25)", dest="epochs", default=25)
    parser.add_argument('-b', '--batch_size', type=int,
                        help="Batch size (default: 32)", dest="batch_size", default=32)
    parser.add_argument("-w", "--workers", type=int,
                        help="Number of workers  (default: 1, if > 1 activate multiprocessing)", dest="workers", default=1)
    parser.add_argument("-dir", "--dir_predict_path", type=str,
                        help="Path to the directory with images", dest="dir_predict_path", default=None)
    args = parser.parse_args()
    if args.train:
        kerasTrain = KerasTrain(epochs=args.epochs, batch_size=args.batch_size,
                                use_multiprocessing=True if args.workers > 1 else False, workers=args.workers)
        kerasTrain.train(
            dict(
                rotation_range=20,
                zoom_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest"
            ),
            modelPath=args.model_path
        )

    if args.img_path is not None:
        model = KerasTrain().loadModel(path=args.model_path)
        model.detect_face_and_predict(args.img_path, f"output-{args.img_path}")

    if args.dir_predict_path is not None:
        model = KerasTrain().loadModel(path=args.model_path)
        model.predictDirectory(dirPath=args.dir_predict_path)
