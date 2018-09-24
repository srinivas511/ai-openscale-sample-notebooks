"""
A very simple Boston data house price prediction.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from keras.losses import mean_squared_error
from keras.callbacks import TensorBoard
import numpy as np

FLAGS = None

def main(_):
  if (FLAGS.result_dir[0] == '$'):
    RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
  else:
    RESULT_DIR = FLAGS.result_dir
  model_path = os.path.join(RESULT_DIR, "MODEL")
  if (FLAGS.data_dir[0] == '$'):
    DATA_DIR = os.environ[FLAGS.data_dir[1:]]
  else:
    DATA_DIR = FLAGS.data_dir
  if (FLAGS.log_dir[0] == '$'):
    LOG_DIR = os.environ["JOB_STATE_DIR"]+"/logs/tb/train"
  else:
    LOG_DIR = os.path.join(FLAGS.log_dir, "logs", "tb", "train")

  print(LOG_DIR)
  import tensorflow as tf
  tf.gfile.MakeDirs(LOG_DIR)

  train_features_file = os.path.join(DATA_DIR, FLAGS.train_features_file)
  train_target_file = os.path.join(DATA_DIR, FLAGS.train_target_file)
  test_features_file = os.path.join(DATA_DIR, FLAGS.test_features_file)
  test_target_file = os.path.join(DATA_DIR, FLAGS.test_target_file)

  # Import data
  print("Read data...")
  class DataSets(object):
      pass
  boston = DataSets()
  
  boston.x_train = np.load(train_features_file)
  boston.y_train = np.load(train_target_file)
  boston.x_test = np.load(test_features_file)
  boston.y_test = np.load(test_target_file)
  
  # Create the model
  print("Create model...")
  model = Sequential()
  model.add(Dense(26, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_dim=13))
  model.add(Dense(1, activation='linear'))
  model.compile(optimizer='rmsprop', loss=mean_squared_error)
  
  # Enable model monitoring
  tb = TensorBoard(log_dir=LOG_DIR, histogram_freq=1, write_graph=False, write_images=False)
  
  # Train
  print("Train model...")
  model.fit(boston.x_train, boston.y_train, epochs=FLAGS.epochs_iters, batch_size=FLAGS.batch_size, validation_data=(boston.x_test, boston.y_test), callbacks=[tb], verbose=False)

  # Save trained model
  print("Saving trained model...")

  os.mkdir(os.path.join(RESULT_DIR, 'model'))
  model.save(os.path.join(RESULT_DIR, 'model', 'boston_model.h5'))
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--log_dir', type=str, default='$LOG_DIR', help='Directory with logs')
  parser.add_argument('--train_features_file', type=str, default='features_train.npy', help='File name for train features')
  parser.add_argument('--train_target_file', type=str, default='target_train.npy', help='File name for train target')
  parser.add_argument('--test_features_file', type=str, default='features_test.npy', help='File name for test features')
  parser.add_argument('--test_target_file', type=str, default='target_test.npy', help='File name for test target')
  parser.add_argument('--epochs_iters', type=int, default=20, help='Number of training epochs')
  parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
  main(FLAGS)
