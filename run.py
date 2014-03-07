import subprocess

yaml_template = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:timitlong.TIMIT {
                        start: %(start)s,
                        stop: %(stop)s,
                        window: %(window)s,
                        frame_width: %(fw)d
                    },
    model: !obj:mlp.MLP {
               input_space: !obj:pylearn2.space.Conv2DSpace {
                                shape: [%(shape)i, 1],
                                num_channels: 1
                            },
               layers: [
                        !obj:mlp.ConvRectifiedLinear {
                            layer_name: 'h0',
                            output_channels: %(c0)d,
                            irange: %(ir)f,
                            init_bias: %(ib)f,
                            kernel_shape: [%(k0)d, 1],
                            pool_shape: [1, 1],
                            pool_stride: [1, 1],
                            pool_type: 'max',
                            max_kernel_norm: %(mkn)f
                        },
                        !obj:mlp.ConvRectifiedLinear {
                            layer_name: 'h1',
                            output_channels: 1,
                            irange: %(ir)f,
                            init_bias: %(ib)f,
                            kernel_shape: [1, 1],
                            pool_shape: [1, 1],
                            pool_stride: [1, 1],
                            pool_type: 'max',
                            max_kernel_norm: %(mkn)f,
                            left_slope: 1
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
                   cost: !obj:mlp.Default {},
                   termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
                                              max_epochs: %(max_epochs)i
                                          }
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


constants = {
'c0': 64,
'k0': 101,
'ir': 0.05,
'mkn': 3000000.0,
'start': 43000,
'stop': 45000,
'window': 4000,
'max_epochs':1000
}

# Not Implemented...
deterministic_constants = {
'fw': 100, # = k0-1
'shape': 2000, # = stop-start
'output_length': 1900 # = stop-start-frame_width
}

all_constants = dict(constants.items() + deterministic_constants.items()) 

hparam_sets = []

for i in range(20):
    d = {'ib': 0.0}
    d['lr'] = .000001 * 1.25**i
    hparam_sets.append(d)

# add automatic experiment logging (keep a text file with all yaml/pkl filenames (and another file with details)... the first file will allow us to write scripts to efficiently generate plots or whatever we need to see for every result

# We'd like to be able to run a lot of experiments and process the results automatically.

for hparams in hparam_sets:
    # set the save_path based on chosen hyperparameter set
    save_path = '/data/lisa/exp/kruegerd/ift6266project_results/'
    keys = hparams.keys()
    path = 'run'
    for k in keys:
        path = path+'_'+str(k)+'='+str(hparams[k])
    save_path_best = save_path + path + '_best.pkl'
    save_path = save_path + path + '.pkl'
    print save_path
    subs = dict(hparams.items() + all_constants.items() + {'save_path': save_path, 'save_path_best': save_path_best}.items())
    yaml_str = yaml_template % subs
    # create a new yaml file for the experiment with this set of hyperparameters
    # overwrites existing file!
    yaml_file = open(path+'.yaml', 'w')
    yaml_file.write(yaml_str) 
    yaml_file.close()
    # run experiment
    subprocess.call(['./train.py', yaml_file.name])


