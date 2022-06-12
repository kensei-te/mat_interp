
import subprocess
import tensorflow as tf
import os
import time
import datetime as dt
import json
import optuna
import joblib
import sys
import keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

import configparser
config = configparser.ConfigParser()
config.read('./config.ini')

mysqlconf = 'MySQL'
user_name = config.get(mysqlconf, 'user')
pass_word = config.get(mysqlconf, 'password')
host_name = config.get(mysqlconf, 'host')



def main(wdir, renew_t):
  path = './utils/learn_child.py'
  command = ["python", path, wdir]
  procs=[]

  def run_learning():
    proc = subprocess.Popen(command)
    return proc

  K.clear_session()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  #read settings
  path_setting = os.path.join(wdir, "NNsetting.json")
  with open(path_setting) as f:
      NNsetting = json.load(f)
  pal_n = NNsetting["n_cores"]
  renew = "True"
  study_name = "optuna_NNtemp"
  storage = 'mysql+pymysql://' + user_name + ':' + pass_word + '@' + host_name + '/Mat_interp'
  print(" --- learning starts--- ")
  if str(renew) == str(renew_t):# if true, it deletes existing study
      try:
          optuna.delete_study(study_name=study_name, storage=storage)
          print(study_name, " cleaned")
      except(KeyError):
          pass
  else:# if false, it does not delete existing study and continues
    pass
  start_time = time.strftime('%Y/%m/%d %H:%M:%S')
  for i in range(pal_n):
      proc = run_learning()
      procs.append(proc)
  
  for proc in procs:
      proc.communicate()

  study = optuna.load_study(study_name = study_name, storage = storage)
  path_study = os.path.join(wdir, "study.pkl")
  joblib.dump(study, path_study)
  fin_time = time.strftime('%Y/%m/%d %H:%M:%S')

  t_start = dt.datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
  t_fin = dt.datetime.strptime(fin_time, '%Y/%m/%d %H:%M:%S')
  t_elapsed = t_fin - t_start

  print("study_started:", start_time)
  print("study_finished:", fin_time)
  print("elapsed time:", t_elapsed)
  print("study done!")
  print("study done!")
  print("study done!")
  # import load_study
  # load_study.main(wdir)


    











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
#   print(wdir)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])