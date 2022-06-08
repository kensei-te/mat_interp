import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.integration import TFKerasPruningCallback

import tensorflow as tf

import os
import sys

import json

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow_addons.metrics import RSquare


os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import configparser
config = configparser.ConfigParser()
config.read('./config.ini')

mysqlconf = 'MySQL'
user_name = config.get(mysqlconf, 'user')
pass_word = config.get(mysqlconf, 'password')
host_name = config.get(mysqlconf, 'host')


def main(wdir):
  K.clear_session()
  #read settings
  path_setting = os.path.join(wdir, "NNsetting.json")
  with open(path_setting) as f:
      NNsetting = json.load(f)

  #load train data
  path_train = os.path.join(wdir, "train_df.csv")
  train_df = pd.read_csv(path_train)
  X_train = train_df.loc[:,[NNsetting["X1_name"], NNsetting["X2_name"]]].to_numpy()
  y_train = train_df.loc[:,[NNsetting["Y_name"]]].to_numpy()

  normalizer = Normalization()
  normalizer.adapt(X_train)
  layers = [normalizer]

  mean_y_orig = NNsetting["mean_y_orig"]
  sdev_y_orig = NNsetting["sdev_y_orig"]

  solver = NNsetting["solver"]


  def create_model(trial) -> float:        
    # below is the search hyperparam area
    n_layer = trial.suggest_int("n_layer", 3, 10)
    n_node = trial.suggest_int("n_node", 50, 200)
    act = "relu"
    lr_ini = trial.suggest_loguniform("lr_ini", 0.0005, 0.005)
    lr_val = [lr_ini, lr_ini/5]
    border = [100]
    lr_fn = PiecewiseConstantDecay(border, lr_val)
    if solver == "SGD":
        opt = SGD(learning_rate = lr_fn)
    elif solver == "NAG":
        opt = SGD(learning_rate = lr_fn, nesterov = True, momentum = NNsetting["momentum"])
    elif solver == "RMSprop":
        opt = RMSprop(learning_rate = lr_fn)
    elif solver == "Adam":
        opt = Adam(learning_rate = lr_fn)

    for i in range(n_layer):
      layers.append(Dense(n_node, activation=act, name=str(trial.number*100)+str(i)))
    layers.append(Dense(1,name="temp"+str(trial.number)))

    model = keras.Sequential(layers)
    model.compile(loss=mse, optimizer=opt, metrics=[RSquare(dtype=tf.float32, y_shape=(1,))])
    return model       
    

  def objective(trial):

    tf.keras.backend.clear_session()

    y_train_s = (y_train-mean_y_orig)/sdev_y_orig

    model = create_model(trial)
    if len(X_train)<256:  
      batch = trial.suggest_categorical("batch", [32,64,128])
    elif len(X_train)<1024:
      batch = trial.suggest_categorical("batch", [64,128,256])
    elif len(X_train)<4096:
      batch = trial.suggest_categorical("batch", [128,256,512])
    else:
      batch = trial.suggest_categorical("batch", [256,512,1024])
    epo = 500
    trial_n = trial.number
    weightp = "weighttemp_"+str(trial_n)+".h5"
    weightpath = os.path.join(wdir, weightp)
    prunecall = TFKerasPruningCallback(trial, "r_square")
    checkpoint = ModelCheckpoint(filepath = weightpath,
                          monitor = 'loss',
                          save_best_only = True)
    Earlycall = EarlyStopping(monitor='loss',
                              min_delta=1.0e-10,
                              restore_best_weights=True,
                              patience=20)

    history = model.fit(X_train, y_train_s, batch_size=batch,
              verbose=0, epochs=epo,
              callbacks=[prunecall, checkpoint, Earlycall])
    histname = "histtrial_"+str(trial_n)+".csv"
    path_hist = os.path.join(wdir, histname)
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    # save to csv: 
    hist_csv_file = path_hist
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    r = history.history["r_square"]
    rmax = max(r)
    ind = r.index(rmax)
    l = history.history["loss"][ind]
    trialname = "modeltrial_"+str(trial_n)
    path_trial = os.path.join(wdir, trialname)

    model.load_weights =(weightpath)
    model.save(path_trial)

    K.clear_session()
    return rmax

  study_name = "optuna_NNtemp" 
  storage = 'mysql+pymysql://' + user_name + ':' + pass_word + '@' + host_name + '/Mat_interp'
  n_trials = round(NNsetting["n_trial"] / NNsetting["n_cores"])
  sampler = TPESampler()
  pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=50, reduction_factor=2)
  n_trials = round(NNsetting["n_trial"] / NNsetting["n_cores"])
  study = optuna.create_study(
    study_name = study_name,
    storage = storage,
    sampler = sampler,
    pruner = pruner,
    direction = "maximize",
    load_if_exists = True)
  study.optimize(objective, n_trials=n_trials)
    











class CustomEarlyStopping(tf.keras.callbacks.Callback):
  """
  -- MODIFIED FROM ORIGINAL KERAS by P. B. Castro

  Stop training when a monitored metric has stopped improving.
  Assuming the goal of a training is to minimize the loss. With this, the
  metric to be monitored would be `'loss'`, and mode would be `'min'`. A
  `model.fit()` training loop will check at end of every epoch whether
  the loss is no longer decreasing, considering the `min_delta` and
  `patience` if applicable. Once it's found no longer decreasing,
  `model.stop_training` is marked True and the training terminates.
  The quantity to be monitored needs to be available in `logs` dict.
  To make it so, pass the loss or metrics at `model.compile()`.
  Args:
    monitor: Quantity to be monitored.
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    patience: Number of epochs with no improvement
        after which training will be stopped.
    verbose: verbosity mode.
    mode: One of `{"auto", "min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
    restore_best_weights: Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used. An epoch will be restored regardless
        of the performance relative to the `baseline`. If no epoch
        improves on `baseline`, training will run for `patience`
        epochs and restore weights from the best epoch in that set.
  Example:
  >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
  >>> # This callback will stop the training when there is no improvement in
  >>> # the loss for three consecutive epochs.
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=10, batch_size=1, callbacks=[callback],
  ...                     verbose=0)
  >>> len(history.history['loss'])  # Only 4 epochs are run.
  4
  """

  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None,
               restore_best_weights=False,
               threshold = None):
    super(CustomEarlyStopping, self).__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.restore_best_weights = restore_best_weights
    self.best_weights = None
    self.threshold = threshold

    if mode not in ['auto', 'min', 'max']:
      logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor:
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    self.best_weights = None

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if np.isnan(current):
        # Stop if any nan
        self.wait = 9999
    if current is None:
      return
    if self.restore_best_weights and self.best_weights is None:
      # Restore the weights after first epoch if no progress is ever made.
      self.best_weights = self.model.get_weights()
    if self.threshold is not None:
        # This method if "spikes" we reset patience again huh
        if current > self.threshold:
            self.wait = 0
        else:
            self.wait += 1
    else:
        self.wait += 1
    if self._is_improvement(current, self.best):
      self.best = current
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
      # Only restart wait if we beat both the baseline and our previous best.
      elif self.baseline is None or self._is_improvement(current, self.baseline):
        self.wait = 0

    if self.wait >= self.patience:
      self.stopped_epoch = epoch
      self.model.stop_training = True
      if self.restore_best_weights and self.best_weights is not None:
        if self.verbose > 0:
          print('Restoring model weights from the end of the best epoch.')
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    if monitor_value is None:
      logging.warning('Early stopping conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))
    return monitor_value

  def _is_improvement(self, monitor_value, reference_value):
    return self.monitor_op(monitor_value - self.min_delta, reference_value)


# def test(wdir):
  # print(wdir)

if __name__ == '__main__':
    main(sys.argv[1])