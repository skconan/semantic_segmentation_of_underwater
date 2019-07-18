class MyCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
    
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print('Training: epoch {} ends at {}'.format(epoch, time.time() - self.start_time))
        if epoch % 20 == 0:
            model_path = result_dir + "/model"
            model_list = get_file_path(model_path)
            if len(model_list) > 5:
                model_list = sorted(model_list, reverse=True)
                for m_path in model_list[5:]:
                    os.remove(m_path)
                    print("remove " + m_path)
            preds = self.model.predict(x_test)
            for i in range(10):
                preds_0 = preds[i] * 255.
       
                preds_0 = np.uint8(preds_0.reshape(img_row, img_col, 3))
                preds_0 = cv.resize(preds_0.copy(), (484,304))
                x_test_0 = x_test[i] * 255.
         
                x_test_0 = np.uint8(x_test_0.reshape(img_row, img_col, 3))
                x_test_0 = cv.resize(x_test_0.copy(), (484,304))
                plt.imshow(x_test_0)
                plt.savefig(result_dir + "/predict_result/" + str(epoch) + "_" + str(i) + "_a _test.jpg")
                plt.close()
                plt.imshow(preds_0)
                plt.savefig(result_dir + "/predict_result/" + str(epoch) + "_" + str(i) + "_b_pred.jpg")
                plt.close()