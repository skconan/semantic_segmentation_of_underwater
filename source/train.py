import time
from utilities import *
from model import Autoencoder


def main():
    project_dir = "home/sk/senior_project"
    dataset_dir = project_dir + "/dataset"

    img_dir = dataset_dir + "/images"
    target_dir = dataset_dir + "/groundTruth_seg_train"
    test_dir = dataset_dir + "/groundTruth_seg_test"

    result_dir = project_dir + "/cnn_ae_" + str(time.time()).split(".")[0]
    model_dir = result_dir + "/model"
    pred_dir = result_dir + "/predict_result"

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        os.mkdir(model_dir)
        os.mkdir(pred_dir)

    TRAIN_IMAGES = []
    TRAIN_TARGET_IMAGES = get_file_path(target_dir)

    VAL_IMAGES = []
    VAL_TARGET_IMAGES = get_file_path(test_dir)

    for train_target_path in TRAIN_TARGET_IMAGES:
        name = get_file_name(train_target_path)
        img_path = img_dir + "/" + name + ".jpg"
        if not os.path.exists(img_path):
            continue
        TRAIN_IMAGES.append(img_path)


    for val_target_path in VAL_TARGET_IMAGES:
        name = get_file_name(val_target_path)
        img_path = img_dir + "/" + name + ".jpg"
        if not os.path.exists(img_path):
            continue
        VAL_IMAGES.append(img_path)


    img_cols = 256
    img_rows = 256

    img_cols_result = 484
    img_rows_result = 304

    x_train = load_image(TRAIN_IMAGES)
    y_train = load_image(TRAIN_TARGET_IMAGES)
    x_val = load_image(VAL_IMAGES)
    y_val = load_image(VAL_TARGET_IMAGES)

    ae = Autoencoder(model_dir=model_dir, pred_dir=pred_dir)
    ae.train_model(x_train, y_train, x_val, y_val, epochs=1000, batch_size=10)

    
if __name__ == "__main__":
    main()