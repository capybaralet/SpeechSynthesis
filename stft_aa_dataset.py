"""
A Pylearn2 Dataset object for accessing TIMIT with all the preprocessing that I want
"""
__authors__ = 'David Krueger'
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["David Krueger"]
__license__ = "3-clause BSD"
__maintainer__ = "David Krueger"

############

import numpy
np = numpy
from numpy import array as A

from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace
#from stft import stft
import scipy
from segmentaxis import segment_axis

def stft(x, fs, framesz, hop):
    """
     x - signal
     fs - sample rate
     framesz - frame size
     hop - hop size (frame size = overlap + hop size)
    """
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

class AA(DenseDesignMatrix):
    """"""

    def __init__(self, 
                 which_set='train',
                 seq_len=3010,
                 frame_size=320,
                 hop_size=32,
                 axes=('b', 0, 1, 'c'),
                 noutput_frames=1,
                 ninput_frames=9,
                 overlap=False, # not implemented yet
                 preprocessing=True):

        self.__dict__.update(locals())
        del self.self

        # cut sequences down to seq_len
        dat = np.load('/data/lisa_ubi/speech/onomatopoeia/dataset/per_phone_timit/wav_aa.npy')
        lengths = [len(i) for i in dat]
        daat = A([A(dat[i][:seq_len]) for i in range(len(dat)) if lengths[i] > seq_len])
        if preprocessing:
            self.mean = np.mean(daat)
            daat -= self.mean
            self.std = np.std(daat)
            daat /= self.std

        # Convert to Spectral
        daat = A([np.abs(stft(ex,1,frame_size,hop_size)).flatten() for ex in daat])
        #print daat.shape
        #aat = A([arr.reshape(len(arr),-1) for arr in daat])
        print daat.shape
        # Stride and flatten
        #print ninput_frames*frame_size
        daat = segment_axis(daat, ninput_frames*frame_size, (ninput_frames-1)*frame_size, axis=1)
        print daat.shape

        if which_set == 'train':
            daat = daat[:int(.8*len(daat))]

        if which_set == 'valid':
            daat = daat[int(.8*len(daat)):int(.9*len(daat))]

        if which_set == 'test':
            daat = daat[int(.9*len(daat)):]

        features = daat[:,:-noutput_frames,:].reshape(-1,frame_size*ninput_frames)
        targets = daat[:,noutput_frames:,-frame_size*noutput_frames:].reshape(-1,frame_size*noutput_frames)

        # FIXME BELOW HERE for CNNs
        #--------------------------

        IMAGES_SHAPE = features.shape + (1,)
        print targets.shape
        print features.shape

        X, y = features, targets
        view_converter = DefaultViewConverter(shape=IMAGES_SHAPE, axes=axes)
        super(AA, self).__init__(X=X, y=y, view_converter=view_converter)


