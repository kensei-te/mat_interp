import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow.keras.backend as K
from tensorflow_addons.metrics import RSquare

import os

import json
import joblib
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

K.clear_session()

def main(wdir):
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #load fixed hyperparameters in NNsetting.json
    path_setting = os.path.join(wdir, "NNsetting.json")
    with open(path_setting) as f:
        NNsetting = json.load(f)

    #load train_data from train_df
    path_train = os.path.join(wdir, "train_df.csv")
    train_df = pd.read_csv(path_train)
    X_train = train_df.loc[:,[NNsetting["X1_name"], NNsetting["X2_name"]]].to_numpy()
    y_train = train_df.loc[:,[NNsetting["Y_name"]]].to_numpy()
    normalizer = Normalization()
    normalizer.adapt(X_train)

    #load optuna study from study.pkl
    path_study = os.path.join(wdir, "study.pkl")
    study = joblib.load(path_study)
    best_trial_n = study.best_trial.number

    print("loading best trial_",best_trial_n)
    model_n = "modeltrial_" + str(best_trial_n)
    path_model = os.path.join(wdir, model_n)
    
    loaded_model = tf.keras.models.load_model(path_model)

    df_temp1 = []
    X2_name = NNsetting["X2_name"]
    first_X2 = train_df[NNsetting["X2_name"]][0]
    X1_vals = train_df[train_df[X2_name] == first_X2].loc[:,NNsetting["X1_name"]]
    X2list = train_df[X2_name].unique()
    for X2 in X2list:
        df_temp2 = pd.DataFrame(data=X1_vals,
                                columns=[NNsetting["X1_name"]])
        df_temp2[NNsetting["X2_name"]] = X2
        features = np.array(df_temp2.loc[:, [NNsetting["X1_name"],NNsetting["X2_name"]]])
        df_temp2[NNsetting["Y_name"]] = loaded_model.predict(features)
        df_temp1.append(df_temp2)
    df_fitted = pd.concat(df_temp1)

    mean_y_orig = NNsetting["mean_y_orig"]
    sdev_y_orig = NNsetting["sdev_y_orig"]
    df_fitted[NNsetting["Y_name"]] = (df_fitted[NNsetting["Y_name"]] * sdev_y_orig) + mean_y_orig

    path_fitted = os.path.join(wdir,"df_fitted.csv")
    df_fitted.to_csv(path_fitted, index=False)

    K.clear_session()
    print("load_study done", time.strftime('%Y/%m/%d %H:%M:%S'))

    import clear_temporary
    clear_temporary.main(wdir)

if __name__ == '__main__':
    main()