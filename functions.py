# Functions and globals for loading and running the Serengeti models

from datetime import datetime
from pathlib import Path
import os
import imghdr
import numpy as np
from fastai import *
from fastai.vision import *
import cv2 as cv
import matplotlib.pyplot as plt


MODEL_PATH = Path("model")
MODEL_FILENAME = "trained_model.pkl"
DATA_PATH = Path("images")

def get_test_images_from_folder(data_path=DATA_PATH):
    test_img_list = [str(Path(data_path)/file) for file in os.listdir(data_path) \
                     if imghdr.what(Path(data_path)/file) in ["jpeg", "png"]]
    print(f"Found {len(test_img_list)} images in folder: {data_path}.")
    return  test_img_list

def load_model(test_img_list):
    print(f"Loading model {MODEL_PATH}/{MODEL_FILENAME}.")
    print(f"Running inference on {len(test_img_list)} images.")
    learn = load_learner(MODEL_PATH, MODEL_FILENAME, test=test_img_list)
    learn.callback_fns=[]
    print(f"Model loaded.")
    return learn

def run_inference(learn):
    inference_start = datetime.now()
    print(f"Starting inference. time={inference_start}")
    preds,y = learn.get_preds(ds_type=DatasetType.Test)
    inference_stop = datetime.now()
    print(f"Inference complete. It took {inference_stop - inference_start}.")
    return preds

def plot_predictions(img_path, pred_dict):
    data = pred_dict
    names = list(data.keys())[::-1]
    values = [round(v,4) for v in list(data.values())[::-1]]

    img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)

    markings_color = (0.667, 0.686, 0.694)
    content_color = (0.106, 0.565, 0.969)
    bg_color = (0.388, 0.416, 0.435)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4), gridspec_kw={'width_ratios': [4, 1]})
    fig.set_facecolor(bg_color)

    ax1.imshow(img)
    ax1.set_axis_off()

    ax2.set_facecolor(bg_color)
    ax2.barh(names, values, color=content_color)
    ax2.set_yticklabels(names, minor=False)
    for i, v in enumerate(values):
        ax2.text(v + 0.01, i, str(v), va='center', color=content_color)
    ax2.tick_params(color=markings_color, labelcolor=markings_color)
    for spine in ax2.spines.values():
        spine.set_edgecolor(markings_color)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    plt.title('Top 5 predictions', color=markings_color)

def print_results(test_img_list, pred_dicts):
    for img, pred_dict in zip(test_img_list, pred_dicts):
        plot_predictions(img, pred_dict)

def run_classification(images_folder=DATA_PATH):
    test_img_list = get_test_images_from_folder()
    learn = load_model(test_img_list)
    preds = run_inference(learn)
    classes = learn.data.classes
    preds_df = pd.DataFrame(
            np.stack(preds),
            index=test_img_list,
            columns=classes,
        )
    pred_dicts=[dict(preds_df.loc[img].nlargest()) for img in test_img_list]
    print_results(test_img_list, pred_dicts)
