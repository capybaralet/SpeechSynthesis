SpeechSynthesis
===============

Speech Synthesis (IFT6266 Project)

This is code for a class project.  There is an accompanying blog here: http://dskspeechsynthesis.wordpress.com/

The most current script to run experiments is run_template.py.  This will allow you to run experiments on CNNs and automatically generate plots of the results automatically.  Read its docstring before use.

You must use my versions of blas.py and cuda.cu instead of those in /theano/sandbox/cuda/.

You must use my version of mlp.py (not pylearn2's).  Some of the yaml files reflect this, some don't.  My code should be compatible with pylearn2 soon.

Both mlp.py and timitlong.py should be in the same directory as the scripts that run experiments.  pylearn2's train.py script should be in this directory as well.

My modifications to mlp.py involve adding an option to use tied biases (tied_b) for CNN kernels/channels.  Without this option, the CNN cannot take variable sized inputs.  


_______________OLD FILES_____________________________________

In run_error.py and run.py, you need to change the save_path. 

runJP.py and run2layerJP.py are scripts designed to mimic some of Jean-Phillipe Raymond's MLPs (http://jpraymond.wordpress.com/)

The following two scripts have been incorporated into run_template.py:
analyze_results.py is a script that will produce plots of prediction results for all pkl files in the working directory.
make_prediction_network.py is like analyze_results, but for the generation task.

