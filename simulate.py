
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow.keras.backend as K
from tensorflow_addons.metrics import RSquare

import os

import json
import joblib
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import configparser
config = configparser.ConfigParser()
config.read('./config.ini')

mysqlconf = 'MySQL'
user_name = config.get(mysqlconf, 'user')
pass_word = config.get(mysqlconf, 'password')
host_name = config.get(mysqlconf, 'host')


K.clear_session()

def main(wdir):
    #load fixed hyperparameters in NNsetting.json
    path_setting = os.path.join(wdir, "NNsetting.json")
    with open(path_setting) as f:
        NNsetting = json.load(f)
    
    #load optuna study from study.pkl
    path_study = os.path.join(wdir, "study.pkl")
    study = joblib.load(path_study)


    #load feature_data from pred_feature.csv
    path_pred_feat = os.path.join(wdir, "pred_feature.csv")
    pred_df = pd.read_csv(path_pred_feat)
    X_pred = pred_df.loc[:,[NNsetting["X1_name"], NNsetting["X2_name"]]].to_numpy()
    
    #reconstruct the best NNmodel found in optuna study
    normalizer = Normalization()
    normalizer.adapt(X_pred)

    best_trial_n = study.best_trial.number
    model_n = "modeltrial_" + str(best_trial_n)
    path_model = os.path.join(wdir, model_n)
    model = keras.models.load_model(path_model)
    # model.summary()

    df_temp1 = []
    X2_name = NNsetting["X2_name"]
    first_X2 = pred_df[NNsetting["X2_name"]][0]
    X1_vals = pred_df[pred_df[X2_name] == first_X2].loc[:,NNsetting["X1_name"]]
    X2list = pred_df[X2_name].unique()
    for X2 in X2list:
        df_temp2 = pd.DataFrame(data=X1_vals,
                                columns=[NNsetting["X1_name"]])
        df_temp2[NNsetting["X2_name"]] = X2
        features = np.array(df_temp2.loc[:, [NNsetting["X1_name"],NNsetting["X2_name"]]])
        df_temp2[NNsetting["Y_name"]] = model.predict(features)
        df_temp1.append(df_temp2)
    df_simulated = pd.concat(df_temp1)

    mean_y_orig = NNsetting["mean_y_orig"]
    sdev_y_orig = NNsetting["sdev_y_orig"]
    df_simulated[NNsetting["Y_name"]] = (df_simulated[NNsetting["Y_name"]]* sdev_y_orig) + mean_y_orig

    path_sim = os.path.join(wdir, "df_simulated.csv")
    df_simulated.to_csv(path_sim, index=False)

    print("simulation done", time.strftime('%Y/%m/%d %H:%M:%S'))
    K.clear_session()

if __name__ == '__main__':
    main()