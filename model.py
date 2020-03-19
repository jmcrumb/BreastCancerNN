"""
Neural Network designed to classify excised breast tissue in breast cancer patients by 
mapping electrical impedance measurements to classifications of breast cancer.

Author: Maxon Crumb
Last Modified: 02/27/2020

Data Source: @url http://archive.ics.uci.edu/ml/datasets/Breast+Tissue#
Project Idea: @url http://neuroph.sourceforge.net/tutorials/BreastTissueClassification/BreastTissueClassification.html 
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
from pandas import ExcelFile
import numpy as np
import os

class AccuracyCallback(tf.keras.callbacks.Callback):
    """Keras Callback which stops the training of the neural network when it exceeds 
    a constant threshold in the range of [0,1]."""

    #accuracy threshold
    __acc_threshold__ = 0.98

    def on_epoch_end(self, epoch, logs = {}):
        """Checks end accuracy level at the end of each epoch.  Will terminate training if 
        constant threshold exceeded"""

        if(logs.get('acc') > self.__acc_threshold__):
            print("\n\n", self.__acc_threshold__ * 100,"% threshold reached.  Ending training session.\n")
            self.model.stop_training = True

def build_model(model_filepath):
    """Constructs tensorflow model.  If model_filepath is None, this function creates a new model with
    randomly initiated weights.  Otherwise, will load previously built model from model_filepath.
    Returns the model."""

    if model_filepath is None:
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(units = 64, input_shape = [9], activation = tf.nn.sigmoid),
                                        tf.keras.layers.Dense(units = 6, name = 'result_layer', activation = tf.nn.softmax)])
        model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    else:
        try:
            model = tf.keras.models.load_model(model_filepath)
        except OSError:
            print('Error: file not recognized.  Please enter again or ^C to quit.\n')
            main()
    return model

def train(model, data_filepath, label_filepath, checkpoint_filepath):
    """Trains the model until constant number of epochs or until the accuracy threshold (see AccuracyCallback)
    is reached."""

    acc_callback = AccuracyCallback()
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath,
                                                             save_weights_only = True,
                                                             verbose = 1)
    training_data, training_labels = read_in_data(data_filepath, label_filepath)
    model.fit(training_data, training_labels, epochs = 40000, callbacks = [acc_callback, checkpoint_callback])

def read_in_data(data_filepath, label_filepath):
    """Reads in training data and label data from a comma seperated value (*.csv) filepath and label_filepath 
    respectively.  The rows of these two files shoud correlate.  The data should be of the format
    I0,PA500,HFS,DA,Area,A/DA,Max IP,DR,P 
    which maps to one of the categories, represented in label data, {car (carcinoma), fad (fibro-adenoma), mas (mastopathy), gla (glandular), 
    con (connective), adi (adipose)}.
    """

    print('Reading in data...\n')
    excel_data = pd.read_csv(data_filepath, delimiter = ',', skipinitialspace = True, 
                             header = None, dtype = float)
    excel_labels = pd.read_csv(label_filepath, delimiter = '\n', skipinitialspace = True, 
                               header = None, dtype = str)
    class_converter = {'car' : 0, "fad" : 1, "mas" : 2, "gla" : 3, "con" : 4, "adi" : 4}
    labels = []
    for lab in excel_labels[0]:
        if lab not in class_converter:
            raise Exception("Error: Word not recognized: " + lab)
        current_set = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        current_set[class_converter[lab]] = 1.0
        labels.append(current_set)
    return np.array(excel_data), np.array(labels)

def load_model():
    """UI prompt for loading a model filepath"""
    txt = input("Enter model filepath or n to train new model: ")
    if txt == 'n':
        return None
    else:
        return txt

def main():
    #Preset filepaths for training dataset
    data_filepath = '/Users/MaxonCrumb/Documents/vsWorkspace/TensorFlow/BreastCancer/data/bscan_data.csv'
    label_filepath = '/Users/MaxonCrumb/Documents/vsWorkspace/TensorFlow/BreastCancer/data/bscan_label.csv'
    #Preset filepaths for testing datasets
    test_data_filepath = '/Users/MaxonCrumb/Documents/vsWorkspace/TensorFlow/BreastCancer/data/bscan_tdata.csv'
    test_label_filepath = '/Users/MaxonCrumb/Documents/vsWorkspace/TensorFlow/BreastCancer/data/bscan_tlabel.csv'
    #Preset filepath for checkpoint log
    checkpoint_filepath = '/Users/MaxonCrumb/Documents/vsWorkspace/TensorFlow/BreastCancer/model_log/checkpoints/cp.ckpt'
    #Preset filepath for model archive
    archive_filepath = '/Users/MaxonCrumb/Documents/vsWorkspace/TensorFlow/BreastCancer/model_log/model_archive/modelRMS_v2.h5'
    load_filepath = load_model()
    if load_filepath != None:
        archive_filepath = load_filepath
    model = build_model(load_filepath)
    try:
        train(model, data_filepath, label_filepath, checkpoint_filepath)
    except KeyboardInterrupt:
        #Catches keyboard interrupts and recovers so model can be saved at stopping point
        print('\n\nKeyboard Interrupt Caught: Closing session.')
        pass
    
    #Test data
    test_data, test_labels = read_in_data(test_data_filepath, test_label_filepath)
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print('Test Loss: {}\nTest Accuracy: {}'.format(test_loss, test_accuracy))
    model.summary()

    print('\nSaving model...\n')
    model.save(archive_filepath)
    return

if __name__ == '__main__': main()


    