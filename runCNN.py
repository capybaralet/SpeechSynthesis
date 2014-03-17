import subprocess
import os.path

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
                            max_kernel_norm: %(mkn)f,
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
                                    frame_width: %(fw)i
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
'c0': 100,
'c1': 100,
'k0': 51,
'k1': 50,
'ir': 0.05,
'mkn': 3000000.0,
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

# add automatic experiment logging (keep a text file with all yaml/pkl
# filenames and details).  How big are pkl files? does it make sense to keep
# them around? I can make a file automatically for each experimetn and put
# results there.

# We'd like to be able to run a lot of experiments and process the results automatically.
# I can probably just wrap my other scripts into functions and import them...
# or copy paste it...

for hparams in hparam_sets:
    save_path = '/data/lisa/exp/kruegerd/ift6266project_results/'
    # set the save_path based on chosen hyperparameter set
    keys = hparams.keys()
    path = os.path.basename(__file__) # filename
    for k in keys:
        path = path+'_'+str(k)+'='+str(hparams[k])
    save_path_best = save_path + path + '_best.pkl'
    save_path = save_path + path + '.pkl'
    print save_path
    this_runs_params = dict(hparams.items() + all_constants.items() + 
                            {'save_path': save_path, 'save_path_best': save_path_best}.items() )
    yaml_str = yaml_template % this_runs_params
    # overwrites existing file!
    yaml_file = open(path+'.yaml', 'w')
    yaml_file.write(yaml_str) 
    yaml_file.close()
    # run experiment
    subprocess.call(['./train.py', yaml_file.name])


