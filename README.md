SpeechSynthesis
===============
New additions: aa_dataset and stft.py.  You should remove the plotting from istft.py to use it, or check out the version I posted here:
http://stackoverflow.com/questions/2459295/stft-and-istft-in-python



Speech Synthesis (IFT6266 Project)

This is code for a class project.  There is an accompanying blog here: http://dskspeechsynthesis.wordpress.com/

The most current script to run experiments is run_template.py.  This will allow you to run experiments on CNNs and automatically generate plots of the results automatically.  Read its docstring before use.

You need the bleeding edge versions of Theano and Pylearn2.

timitlong.py should be in the same directory as the scripts that run experiments. 

You can ignore the bash scripts.

_______________OLD FILES_____________________________________

The OLD file contains code I'm no longer using.

In run_error.py and run.py, you need to change the save_path. 

runJP.py and run2layerJP.py are scripts designed to mimic some of Jean-Phillipe Raymond's MLPs (http://jpraymond.wordpress.com/)

The following two scripts have been incorporated into run_template.py:
analyze_results.py is a script that will produce plots of prediction results for all pkl files in the working directory.
make_prediction_network.py is like analyze_results, but for the generation task.

