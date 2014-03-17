import subprocess
import os.path

import cPickle
P = cPickle
import numpy as np
import theano
import theano.tensor as T
from theano import function as F
import os
import timitlong
import pylab as plt
import pylearn2.models.mlp as mlp
import mlp
from numpy.random import normal
import scipy.io.wavfile as wav
from pylearn2.space import VectorSpace

""" How to use this script, and what to modify:

    This is meant to be a general purpose script for running experiments for a project in an organized fashion.  It is a WIP (work-in-progress).

    It runs experiments for a fixed network architecture (given by yaml_template) with a given set of hyperparameters (hparam_sets).
    The results of the experiments are stored in: /my_dir/project_name/this_script_name/exp_num/hyperparams_of_a_model.
    "exp_num" is the number of times this script has been called previously (i.e. the number of times the experiment has been ran)
    
    If the experiment is changed, the script name should be as well (for now, would like to automate this).  

    For anyone other than me (David Krueger), change the following paths:
      my_dir = '/data/lisa/exp/kruegerd/' # base directory for experiments and results (yaml, pkl, jpg files)
      train_dir = '/u/kruegerd/repo/pylearn2/pylearn2/scripts/train.py' # directory of pylearn2/scripts/train.py
   
    When starting a new project, change the project_name.

    To configure the network, change the yaml_template, the parameters, and the SCRIPTNAME! (this should be automated, as well...)



    TODO: make it so you don't need to change the name if you make mods (it is detected automatically)...

    TODO: deal with imports, find a place for train.py...

    TODO: check all dependencies, etc. on constants, handle it better... (checkout dataset as well)

    TODO: Improve logging so that it can record more information, such as:
        1. script contents
        2. system information
        3. software version (etc.) information 

    TODO: manage directories (stay in original_dir?)

X   TODO: early_stopping 

"""

#############################################################################
# Parameters for experiments

project_name = "IFT6266/"
my_dir = '/data/lisa/exp/kruegerd/' # base directory for experiments and results (yaml, pkl, jpg files)
train_dir = '/u/kruegerd/repo/pylearn2/pylearn2/scripts/train.py' # directory of pylearn2/scripts/train.py

constants = {
'c0': 100,
'c1': 100,
'k0': 51,
'k1': 50,
'ir': 0.05,
'start': 43000,
'stop': 45148,
'window': 4000,
'max_epochs':500
}

deterministic_constants = {
'shape': constants['stop'] - constants['start'],
'output_length': constants['stop'] - constants['start'] - constants['k0'],
'fw': constants['k0'] + constants['k1'] - 1
}

all_constants = dict(constants.items() + deterministic_constants.items()) 

hparam_sets = []

for i in range(10):
    d = {'ib': 0.0}
    d['lr'] = .000011 * 1.5**i
    hparam_sets.append(d)

# for deterministic pseudo-randomness via passing rng parameter.
# NOT IMPLEMENTED
_default_seed = (17, 2, 946)

frame_width = 100
tl = timitlong.TIMIT(which_set='test', start=43000, stop=45148, window=4000,frame_width=100) # CHECK ME!
X0 = np.array(tl.X[0], dtype='float32')
X0 = X0.reshape(1,len(X0),1,1)
y0 = np.array(tl.y[0], dtype='float32')

#############################################################################
# Set-up for processing results (currently all very project/model specific)

class trained_model:
    def __init__(self, complete_path):
        if os.path.exists(complete_path):
            self.progressMLP = cPickle.load(open(complete_path))
        else:
            self.progressMLP = None
        complete_path_best = os.path.join(complete_path[:-4], '_best.pkl')
        if os.path.exists(complete_path_best):
            self.bestMLP = cPickle.load(open(complete_path_best))
        else:
            self.bestMLP = None

# data: one test sequence (X0)
def generate(data,W1,W2,b1,b2,frame_width,sigma,rng=_default_seed,length=10000):
    """generate a sequence of a given length, using data[:frame_width] as a
       starting point"""
    sequence = data.flatten()
    assert length > frame_width
    output = np.zeros(length)
    output[:frame_width] = sequence[:frame_width]
    for i in range(frame_width,length):
        if i > length - 5: print i
        output[i] = compute_prediction(output[i-frame_width:i],W1,W2,b1,b2,sigma,rng)
    return output

def compute_prediction(data,W1,W2,b1,b2,sigma,rng=_default_seed):
    """generate the next item in a sequence (output of MLP with
       params=W1,W2,b1,b2), using data as input"""
    #print "shapes",W1.shape,W2.shape,b1.shape,b2.shape
    ha = np.dot(W1,data)+b1
    hs = ha*(ha>0)
    oa = np.dot(W2,hs)+b2
    return oa + normal(0,sigma)

#############################################################################
yaml_template = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:timitlong.TIMIT {
                        start: %(start)s,
                        stop: %(stop)s,
                        window: %(window)s,
                        frame_width: %(fw)d
                    },
    model: !obj:pylearn2.models.mlp.MLP {
               input_space: !obj:pylearn2.space.Conv2DSpace {
                                shape: [%(shape)i, 1],
                                num_channels: 1
                            },
               layers: [
                         !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                            layer_name: 'h0',
                            output_channels: %(c0)d,
                            irange: %(ir)f,
                            init_bias: %(ib)f,
                            kernel_shape: [%(k0)d, 1],
                            pool_shape: [1, 1],
                            pool_stride: [1, 1],
                            pool_type: 'max',
                            tied_b: True
                        },
                        !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                            layer_name: 'h1',
                            output_channels: %(c1)d,
                            irange: %(ir)f,
                            init_bias: %(ib)f,
                            kernel_shape: [%(k1)d, 1],
                            pool_shape: [1, 1],
                            pool_stride: [1, 1],
                            pool_type: 'max',
                            tied_b: True
                        },
                        !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                            layer_name: 'h2',
                            output_channels: 1,
                            irange: %(ir)f,
                            init_bias: %(ib)f,
                            kernel_shape: [1, 1],
                            pool_shape: [1, 1],
                            pool_stride: [1, 1],
                            pool_type: 'max',
                            left_slope: 1,
                            tied_b: True
                        }
                       ]
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
                   batch_size: 1,
                   learning_rate: %(lr)f,
                   monitoring_dataset: {
                       'train': *train,
                       'valid': !obj:timitlong.TIMIT {
                                    which_set: 'valid',
                                    start: %(start)s,
                                    stop: %(stop)s,
                                    window: %(window)s,
                                    frame_width: %(fw)i
                                },
                   },
                   cost: !obj:pylearn2.models.mlp.Default {},
                   termination_criterion: !obj:pylearn2.termination_criteria.Or {
                                              criteria: [termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
                                                                                    channel_name: 'valid_objective',
                                                                                    prop_decrease: 0.005,
                                                                                    N: 10
                                                         },
                                                         termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
                                                                                    max_epochs: %(max_epochs)i
                                                         }]
                   },
    },
    save_freq: 1,
    save_path: %(save_path)s,
    extensions: [
                 !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                     channel_name: 'valid_objective',
                     save_path: %(save_path_best)s
                 },
                 !obj:pylearn2.training_algorithms.sgd.MonitorBasedLRAdjuster {
                     channel_name: 'valid_objective'
                 }
                ]
}"""

# results are stored in:
# /my_dir/project_name/this_script_name/exp_num/hyperparams_of_a_model.____(pkl, jpg, etc.)
# a project is a user-defined organizational tool
# an "experiment" is a run of the script

# not sure how to handle moving through directories (maybe just don't)... 
# Ctrl+C/crash seems hard to handle...
original_dir = os.getcwd()

trained_models = []

filename = os.path.basename(__file__)[:-3] #remove '.py'

# make project_dir if it doesn't exist
project_dir = os.path.join(my_dir,project_name)
if not os.path.exists(project_dir): os.makedirs(project_dir)
# make script_dir if it doesn't exist
script_dir = os.path.join(project_dir,filename)
if not os.path.exists(script_dir): os.makedirs(script_dir)
# make txt file recording number of experiments performed with this script
log_path = os.path.join(script_dir,'exp_log.txt')
if not os.path.exists(log_path):
    exp_num = 0
else: 
    for i in open(log_path, 'r'):
        exp_num = int(i) + 1 
log = open(log_path, 'w')
log.write(str(exp_num))
log.close()
# make exp_dir for each experiment
exp_dir = os.path.join(script_dir,str(exp_num))
os.makedirs(exp_dir)
os.chdir(exp_dir)
# make txt file in each subdir recording when a model has finished training
training_log_path = os.path.join(exp_dir,'trained_models.txt')
training_log = open(training_log_path, 'w')
training_log.write('The following models have finished training:')
training_log.close()


#######################################################################################
# Run Experiments
for hparams in hparam_sets:
    path = filename
    save_path = exp_dir
    for k in hparams.keys():
        path = path + '_' + str(k) + '=' + str(hparams[k])
    save_path_best = os.path.join(save_path, path + '_best.pkl')
    save_path = os.path.join(save_path, path + '.pkl')
    print "save_path", save_path
    this_runs_params = dict(hparams.items() + all_constants.items() + 
                            {'save_path': save_path, 'save_path_best': save_path_best}.items() )
    yaml_str = yaml_template % this_runs_params
    # overwrites existing file!
    yaml_file = open(path + '.yaml', 'w')
    yaml_file.write(yaml_str) 
    yaml_file.close()
    # run experiment
    subprocess.call(['/u/kruegerd/repo/pylearn2/pylearn2/scripts/train.py', yaml_file.name]) 
    training_log = open(training_log_path, 'a')
    training_log.write('\n'+yaml_file.name)
    training_log.close()
    print "done training!"
    ####################
    # MAKE PLOTS 
    m = trained_model(save_path) 
    trained_models.append(m) 
    bestM = m.bestMLP 
    X = bestM.get_input_space().make_batch_theano() 
    yhat = bestM.fprop(X) 
    f = F([X], yhat) 
    #print 'figure! - predictions' # plot prediction 
    yhat0 = f(X0).flatten() 
    fig = plt.figure() 
    fig.suptitle(path) 
    plt.subplot(221) 
    plt.plot(yhat0, 'g') 
    plt.plot(y0, 'b') 
    plt.subplot(222) 
    plt.plot(yhat0[:100], 'g') 
    plt.plot(y0[:100], 'b') 
    #print 'figure! - learning curves' # plot learning curves MAKE MORE! 
    M = m.progressMLP 
    train_obj = M.monitor.channels['train_objective'].val_record 
    valid_obj = M.monitor.channels['valid_objective'].val_record 
    plt.subplot(223) 
    plt.plot(train_obj, 'g') 
    plt.subplot(224) 
    plt.plot(valid_obj, 'b') 
    fig.savefig(path[:-4]+'.jpg')
    # Generate from trained model (only handles runJP case currently)
    if path[:5] == 'runJP':
        old_layers = Mbest.layers
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
        sigma = Mbest.monitor.channels['valid_objective'].val_record[-1]**.5
        generated = generate(X0, W1, W2, b1, b2, frame_width, sigma)
        fig = plt.figure()
        plt.subplot(311)
        plt.plot(generated, 'g')
        plt.subplot(312)
        plt.plot(X0.flatten()[:10000], 'b')
        plt.subplot(313)
        plt.plot(generated, 'g')
        plt.plot(X0.flatten()[:10000], 'b')
        savepath = save_path[:-4] + '_generated'
        fig.savefig(savepath + '.jpg')
        wav.write(savepath + '.wav', 16000, generated)

os.chdir(original_dir)

