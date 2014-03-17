import cPickle
P = cPickle
import numpy as np
#import theano
import theano.tensor as T
from theano import function as F
import os
import timitlong
import pylab as plt
from trained_model import trained_model
#import pylearn2.models.mlp as mlp
import mlp
from numpy.random import normal
import scipy.io.wavfile as wav
from pylearn2.space import VectorSpace


# for deterministic pseudo-randomness via passing rng parameter.
# NOT IMPLEMENTED
_default_seed = (17, 2, 946)

"""
Right now, this is a script that works, like analyze_results.py, to create plots for all .pkl files in a directory
This script loads these trained models and uses them to generate sequences.  

This should eventually be a script that transforms trained (variable length) CNNs into single-output CNNs using the same parameters.
We are starting with models with one hidden layer....


PLAN:
    1. We should modify our model to be another class that is truly variable length (but maybe still trains on one length of data at a time?)
    2. Our new model class should, at a minimum, provide information needed to construct the (single-output) new_model.

"""

#########
# THESE FUNCTIONS ARE WIPs (works-in-progress)

# model: MLP
# data: one test sequence (X0)
def generate2(model, data, frame_width, sigma, rng=_default_seed, algorithm='spherical_gaussian'):
# this function is a WIP
    output = np.zeros(10000)
    output[:frame_width] = data.flatten()[:frame_width]
    for i in range(frame_width,10000):
         output[i] = compute_predictions(model,output[i-frame_width:i],sigma,rng,algorithm)
    return output

def compute_predictions(model, data, sigma, rng=_default_seed, algorithm='spherical_gaussian'):
# this function is a WIP
    return fprop(model, data) + normal(0,sigma)

def fprop(model, data):
# this function is a WIP
    l = model.layers
    W1 = 1#l[0].
    b1 = 1
    W2 = 1
    b2 = 1
    ha = np.dot(W1,data)+b1
    hs = ha*(ha>0)
    oa = np.dot(W2,hs)+b2
    return oa

# END WIP functions
########################


# directory = user input
directory = os.getcwd()

#####
# LOAD TEST DATA
# For a given exp_batch, we want a start, stop, frame to be constant (or at least output_length = stop-start-frame_width)
frame_width = 100
tl = timitlong.TIMIT(which_set='test', start=5000, stop=45148, window=4000, frame_width=100) # CHECK ME!
X0 = np.array(tl.X[0], dtype='float32')
X0 = X0.reshape(1,len(X0),1,1)
y0 = np.array(tl.y[0], dtype='float32')


# data: one test sequence (X0)
def generate(data,W1,W2,b1,b2,frame_width,sigma,rng=_default_seed,length=10000):
    """generate a sequence of a given length, using data[:frame_width] as a starting point"""
    sequence = data.flatten()
    assert length > frame_width
    output = np.zeros(length)
    output[:frame_width] = sequence[:frame_width]
    for i in range(frame_width,length):
        if i > length - 5: print i
        output[i] = compute_prediction(output[i-frame_width:i],W1,W2,b1,b2,sigma,rng)
    return output

def compute_prediction(data,W1,W2,b1,b2,sigma,rng=_default_seed):
    """generate the next item in a sequence (output of MLP with params=W1,W2,b1,b2), using data as input"""
    #print "shapes",W1.shape,W2.shape,b1.shape,b2.shape
    ha = np.dot(W1,data)+b1
    hs = ha*(ha>0)
    oa = np.dot(W2,hs)+b2
    return oa + normal(0,sigma)

#####
# load all .pkl files in this directory
paths = os.listdir(directory)
# skip all paths ending in '_best.pkl' so we don't initialize two "trained_model"s per experiment...
pkl_paths = [p for p in paths if p[-4:] == '.pkl' and p[-9:] != '_best.pkl' and p[:4] != 'run2']
trained_models = []
for path in pkl_paths:
    print path
    # Load model
    m = trained_model(path)
    trained_models.append(m)
    M = m.bestMLP
    # Make layers (only handles runJP case currently)
    old_layers = M.layers
    new_layers = []
    l0 = old_layers[0]
    l1 = old_layers[1]
    W1 = np.squeeze(np.array(l0.get_params()[0].eval()))
    W2 = np.squeeze(np.array(l1.get_params()[0].eval()))
    b1 = np.squeeze(np.array(l0.get_params()[1].eval()))
    b2 = np.squeeze(np.array(l1.get_params()[1].eval()))
    if len(b2)>1: b2 = b2[0]
    if len(b1.shape) > 1: b1 = b1[:,0]
    print W1.shape, W2.shape, b1.shape, b2.shape, "shapes"
#    input_space = l0.get_input_space()
#   output_space = l0.get_output_space()
#    c0 = output_space.num_channels
#    k0 = input_space.shape[0] - output_space.shape[0]
#    L0 = mlp.RectifiedLinear(dim=c0, layer_name='L0')
#    L1 = mlp.Linear(dim=1, layer_name='L1')
#    L0.set_input_space(VectorSpace(dim=c0))
#    L1.set_input_space(VectorSpace(dim=k0))
#    L0.output_space = VectorSpace(dim=k0)
#    L1.output_space = VectorSpace(dim=1)
    # Construct and save new model
#    new_model = mlp.MLP(layers=[L0,L1], nvis=k0)
    # Still need to get and set parameters!!!
    sigma = M.monitor.channels['valid_objective'].val_record[-1]
    generated = generate(X0,W1,W2,b1,b2,frame_width,sigma)
    fig = plt.figure()
    plt.subplot(311)
    plt.plot(generated, 'g')
    plt.subplot(312)
    plt.plot(X0.flatten()[:10000], 'b')
    plt.subplot(313)
    plt.plot(generated, 'g')
    plt.plot(X0.flatten()[:10000], 'b')
    savepath = path[:-4]+'_generated'
    fig.savefig(savepath+'.jpg')
    wav.write(savepath+'.wav',16000,generated)












