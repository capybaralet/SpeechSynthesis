import scipy, pylab
import matplotlib.pyplot as plt
import numpy as np

numplots = 5
seq_len = 3010

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

def istft(X, fs, T, hop, j):
    """ T - signal length """
    length = T*fs
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    # calculate the inverse envelope to scale results at the ends.
    xx = scipy.zeros(T*fs)
    w = scipy.hamming(framesamp)
    for i in range(0, len(x)-framesamp, hopsamp):
        xx[i:i+framesamp] += w
    #xx[-(length%hopsamp):] += w[-(length%hopsamp):]
    plt.subplot(numplots,3,3*j+3)
    plt.plot(xx)
    xx = np.maximum(xx, .01)
    plt.plot(xx, 'r')
    return x/xx # right side is still a little messed up...

from aa_dataset import AA

transforms = []
recons = []

X = AA(seq_len = seq_len).X[2]

fig = plt.figure()
fig.suptitle('(abs) Fourier transforms and reconstructions (red)')

for i,framesz in enumerate([160,240,320,480,640]):
    Xt = stft(X, 16000, framesz/16000., .002)
    transforms.append(Xt)
    Xtt = istft(Xt, 16000, seq_len/16000., .002, i) #*32/framesz # 32: make it a parameter!
    recons.append(Xtt)
    plt.subplot(numplots,3,3*i+1)
    plt.plot(X)
    plt.plot(Xtt, 'r')
    plt.subplot(numplots,3,3*i+2)
    plt.imshow(np.abs(Xt).T, origin='lower', aspect='auto', interpolation='nearest')
    print Xt.shape

plt.show()
