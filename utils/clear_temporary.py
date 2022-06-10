
import os
import shutil
import glob
import itertools

import joblib
import time


def main(wdir):

    #load optuna study from study.pkl
    path_study = os.path.join(wdir, "study.pkl")
    study = joblib.load(path_study)
    best_trial_n = study.best_trial.number

    # print("loading best trial_",best_trial_n)
    # model_n = "modeltrial_" + str(best_trial_n)
    # path_model = os.path.join(wdir, model_n)
    # loaded_model = tf.keras.models.load_model(path_model)

    query1 = wdir + "/*modeltrial_*"
    l1 = glob.glob(query1)
    h = list(itertools.filterfalse(lambda x: x.split("modeltrial_")[1] == str(best_trial_n),l1))
    for i, name in enumerate(h):
        shutil.rmtree(h[i])

    query2 = wdir + "/*weighttemp_*"
    l2 = glob.glob(query2)
    for i, name in enumerate(l2):
        os.remove(l2[i])

    print("temporary files deleted", time.strftime('%Y/%m/%d %H:%M:%S'))

if __name__ == '__main__':
    main()