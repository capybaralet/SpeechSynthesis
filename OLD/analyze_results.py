import cPickle
P = cPickle
import numpy as np
import theano
import theano.tensor as T
from theano import function as F
import os
import timitlong
import pylab as plt

"""
@ Plan:
	1) Run experiments in batches.  (exp_batch's)
	2) For each batch of experiments, have a directory.
	3) This script takes a directory as input, or (CURRENTLY) works in the folder it is called from
@ What this script does:
	1) make a table containing all results (like JPs)
	(for each experiment in dir):
X	1) create plots of performance vs. time (epochs? or what...)
X	2) create plot of yhat and y (prediction task)
        3) soon: generate audio wav files for y, yhat
	4) eventually: also do generation/synthesis

might just try adapting this:
http://nbviewer.ipython.org/urls/raw2.github.com/davidtob/UdeMDLCourse/master/synthesis_1_MLP/learn_own_data_samples_serious/Generate.ipynb
"""

class trained_model:
    def __init__(self, complete_path):
        if os.path.exists(complete_path):
            self.progressMLP = cPickle.load(open(complete_path))
        else:
            self.progressMLP = None
        complete_path_best = complete_path[:-4] + '_best.pkl'
        if os.path.exists(complete_path_best):
            self.bestMLP = cPickle.load(open(complete_path_best))
        else:
            self.bestMLP = None

        # When we init a model, we can also load whatever we need to...
        # David Belius saved a txt file with hyperparameter settings...

        self.xsamples = None
        self.ysamples = None
        self.recenter_samples = None
        self.rescaling = None
        self.num_hidden_layer = None
        self.learning_rate = None
        self.regularization = None
        self.regularization_coeff = None
        self.training_diagram = None
        self.generated_waveform = None



# directory = user input
directory = os.getcwd()

#####
# LOAD TEST DATA
# For a given exp_batch, we want a start, stop, frame to be constant (or at least output_length = stop-start-frame_width)
tl = timitlong.TIMIT(which_set='test', start=43000, stop=45148, window=4000, frame_width=100) # CHECK ME!
X0 = np.array(tl.X[0], dtype='float32')
X0 = X0.reshape(1,len(X0),1,1)
y0 = np.array(tl.y[0], dtype='float32')

#####
# load all .pkl files in this directory
paths = os.listdir(directory)
# skip all paths ending in '_best.pkl' so we don't initialize two "trained_model"s per experiment...
pkl_paths = [p for p in paths if p[-4:] == '.pkl' and p[-9:] != '_best.pkl']
trained_models = []
for path in pkl_paths:
    print path
#TO DO: deal with figures better...
    m = trained_model(path)
    trained_models.append(m)
    M = m.bestMLP
    X = M.get_input_space().make_batch_theano()
    yhat = M.fprop(X)
    f = F([X], yhat)
    #print 'figure! - predictions' # plot prediction
    yhat0 = f(X0).flatten()
    fig = plt.figure()
    fig.suptitle(path)
    plt.subplot(321)
    plt.plot(yhat0, 'g')
    plt.plot(y0, 'b')
    plt.subplot(322)
    plt.plot(yhat0[:100], 'g')
    plt.plot(y0[:100], 'b')
    #print 'figure! - learning curves' # plot learning curves MAKE MORE!
    M = m.progressMLP
    train_obj = M.monitor.channels['train_objective'].val_record
    valid_obj = M.monitor.channels['valid_objective'].val_record
    plt.subplot(323)
    plt.plot(train_obj, 'g')
    plt.subplot(324)
    plt.plot(valid_obj, 'b')
    #print 'figure! - predictions' # plot prediction
    yhat = M.fprop(X)
    f = F([X], yhat)
    yhat0 = f(X0).flatten()    
    plt.subplot(325)
    plt.plot(yhat0, 'g')
    plt.plot(y0, 'b')
    plt.subplot(326)
    plt.plot(yhat0[:100], 'g')
    plt.plot(y0[:100], 'b')
    #plt.show()
    fig.savefig(path[:-4]+'.jpg')

#####
# What ELSE to analyze???
#





