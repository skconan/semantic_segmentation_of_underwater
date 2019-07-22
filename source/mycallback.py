import keras
import time
from utilities import *
import matplotlib.pyplot as plt
import cv2 as cv


class MyCallback(keras.callbacks.Callback):
    def __init__(self, x_test, model_dir, pred_dir, img_rows=256, img_cols=256, channels=3, img_rows_result=304, img_cols_result=484):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels

        self.img_rows_result = img_rows_result
        self.img_cols_result = img_cols_result

        self.model_dir = model_dir
        self.pred_dir = pred_dir

        self.x_test = x_test

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def keep_last_models(self, n=5):
        model_list = get_file_path(self.model_dir)
        if len(model_list) > n:
            model_list = sorted(model_list, reverse=True)
            for m_path in model_list[n:]:
                os.remove(m_path)
                print("remove " + m_path)

    def on_epoch_end(self, epoch, logs=None):
        print('Training: epoch {} ends at {}'.format(
            epoch, (time.time() - self.start_time)/60.))
        self.keep_last_models()
        if epoch % 20 == 0:
            preds = self.model.predict(self.x_test)

            for i in range(len(preds)):
                pred = preds[i] * 255.
                pred = np.uint8(pred.reshape(
                    self.img_rows, self.img_cols, self.channels))
                pred = cv.resize(
                    pred.copy(), (self.img_cols_result, self.img_rows_result))

                x_test = self.x_test[i] * 255.
                x_test = np.uint8(x_test.reshape(
                    self.img_rows, self.img_cols, self.channels))
                x_test = cv.resize(
                    x_test.copy(), (self.img_cols_result, self.img_rows_result))

                plt.imshow(x_test)
                plt.savefig(self.pred_dir + "/" +
                            str(epoch) + "_" + str(i) + "_a _test.jpg")
                plt.close()

                plt.imshow(pred)
                plt.savefig(self.pred_dir + "/" +
                            str(epoch) + "_" + str(i) + "_b_pred.jpg")
                plt.close()
