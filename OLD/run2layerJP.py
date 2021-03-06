import subprocess

yaml_template = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:timitlong.TIMIT {
                        start: %(start)s,
                        stop: %(stop)s,
                        window: %(window)s,
                        frame_width: %(k0)d
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
                            max_kernel_norm: %(mkn)f,
                            tied_b: True
                        },
                        !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                            layer_name: 'h1',
                            output_channels: %(c1)d,
                            irange: %(ir)f,
                            init_bias: %(ib)f,
                            kernel_shape: [1, 1],
                            pool_shape: [1, 1],
                            pool_stride: [1, 1],
                            pool_type: 'max',
                            max_kernel_norm: %(mkn)f,
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
                            max_kernel_norm: %(mkn)f,
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
                                    frame_width: %(k0)i
                                },
                   },
                   cost: !obj:pylearn2.models.mlp.Default {},
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
'c0': 300,
'c1': 200,
'k0': 100,
'ir': 0.05,
'mkn': 3000000.0,
'start': 43000,
'stop': 45148,
'window': 4000,
'max_epochs':1500
}

deterministic_constants = {
'shape': constants['stop'] - constants['start'],
'output_length': constants['stop'] - constants['start'] - constants['k0']
}

all_constants = dict(constants.items() + deterministic_constants.items()) 

hparam_sets = []

for i in range(4):
    d = {'ib': 0.0}
    d['lr'] = .0000011 * 4**i
    hparam_sets.append(d)

# add automatic experiment logging (keep a text file with all yaml/pkl filenames (and another file with details)... the first file will allow us to write scripts to efficiently generate plots or whatever we need to see for every result

# We'd like to be able to run a lot of experiments and process the results automatically.

for hparams in hparam_sets:
    # set the save_path based on chosen hyperparameter set
    save_path = '/data/lisa/exp/kruegerd/ift6266project_results/'
    keys = hparams.keys()
    path = 'run2layerJPR'
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


