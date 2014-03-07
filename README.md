SpeechSynthesis
===============

Speech Synthesis (IFT6266 Project)

This is code for a class project.  There is an accompanying blog here: http://dskspeechsynthesis.wordpress.com/

You must use my versions of blas.py and cuda.cu instead of those in /theano/sandbox/cuda/.

You must use my version of mlp.py (not pylearn2's).  Both mlp.py and timitlong.py should be in the same directory as run_error.py.  pylearn2's train.py script should be in this directory as well.

In run_error.py and run.py, you need to change the save_path. 

run.py is the latest script for running experiments.  run_error.py is a similar script that demonstrates an error I get when I try to use more than 64 channels in my ConvRectifiedLinear layers on GPU. 

analyze_results.py is a script that will produce plots of results for all pkl files in the working directory.

