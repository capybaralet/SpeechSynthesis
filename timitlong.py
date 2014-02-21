"""
A Pylearn2 Dataset object for accessing TIMIT with all the preprocessing that I want
"""
__authors__ = 'David Krueger'
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["David Krueger"]
__license__ = "3-clause BSD"
__maintainer__ = "David Krueger"




############

import os
import os.path
import numpy
from pylearn2.utils import serial
import csv
from itertools import izip
import math
import time

import numpy as np

import theano.tensor as T
from theano import function

from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
IMAGES_SHAPE = [45000, 1, 1]

class TIMIT(DenseDesignMatrix):
    """
    A Pylearn2 Dataset object for accessing TIMIT w/preprocessing
    """

    # Mean and standard deviation of the acoustic samples from the whole
    # dataset (train, valid, test).
    _mean = 0.0035805809921434142
    _std = 542.48824133746177

    def __init__(self, 
                 which_set='train',
                 transformer=None,
                 start=0,
                 stop=45000,
                 window=250,
                 frame_width=200,
                 preprocessor=None,
                 fit_preprocessor=False,
                 axes=('b', 0, 1, 'c'),
                 fit_test_preprocessor=False,
                 space_preserving=False):

        self.test_args = locals()
        del self.test_args['self']
        # Load data from disk
        self._load_data(which_set)
        self.frame_width = frame_width
        # Standardize and Slice data
        features = []
        targets = []
        for i, sequence in enumerate(self.raw_wav):
            if len(self.raw_wav[i]) < stop+window and len(self.raw_wav[i]) > stop-1:
                self.raw_wav[i] = (sequence - TIMIT._mean) / TIMIT._std
                self.raw_wav[i] = self.raw_wav[i][start:stop]
                features.append(self.raw_wav[i])
                targets.append(self.raw_wav[i][frame_width:stop])
        features = numpy.array(features)
        targets = numpy.array(targets)
        self.raw_wav = features
        #self.data = features, targets
        print "self.raw_wav.shape, self.raw_wav[0].shape = ", self.raw_wav.shape, self.raw_wav[0].shape
        #self.num_examples = len(self.raw_wav)


        X, y = features, targets
        view_converter = DefaultViewConverter(shape=IMAGES_SHAPE, axes=axes)
        super(TIMIT, self).__init__(X=X, y=y, view_converter=view_converter)




    def _load_data(self, which_set):
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")
        # Create file paths
        timit_base_path = os.path.join(os.environ["PYLEARN2_DATA_PATH"],
                                       "timit/readable")
        raw_wav_path = os.path.join(timit_base_path, which_set + "_x_raw.npy")
        # Load data. 
        self.raw_wav = serial.load(raw_wav_path)
