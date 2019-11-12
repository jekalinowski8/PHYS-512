#Jonathan Kalinowski - PHYS 512 Asst 4
import simple_read_ligo
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as nd
import os

directory = '/home/2019/jkalin3/PHYS512/phys512_jk/asst4'
os.chdir(directory)

def matched_filter(data,template,dt,plot=False,a=5):
    window = np.hamming(len(data)) #Hamming window for data/template
    fdata = np.fft.rfft(data*window) #FFT of strain
    power = fdata*np.conjugate(fdata) #Power spectrum
    ftemp = np.fft.rfft(template*window) #FFT of template
    freqs = np.fft.rfftfreq(len(data),dt) #Frequencies
    #Smooth data w/ Gaussian filter for noise model, standard deviation
    #is 5 (I played around with a few different values)
    nn = nd.gaussian_filter1d(np.real(power),a) 
    n = np.sqrt(nn) #sqrt of noise matrix
    wfdata = fdata/n #whitened data
    wfdata[np.logical_or(freqs<300,freqs > 1700)] = 0 #bandpass
    wftemp = ftemp/n #whitened template
    fm = wftemp*wfdata #matched filter in Fourier
    num = np.fft.irfft(fm,len(fm)) #matched filter
    t = range(len(data))*dt #time values
    wdata = np.fft.irfft(wfdata, len(wfdata))
    den = wdata**2
    return den
    m = num/den
    SNR = m*np.sqrt(den)
    if plot:
        plt.plot(t,m)
    return t,m, SNR

#First event
template_name='GW150914_4_template.hdf5'
th,tl=simple_read_ligo.read_template(template_name)
fname='H-H1_LOSC_4_V2-1126259446-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
t,m,SNRH=matched_filter(strain,th,dt)
fname='L-L1_LOSC_4_V2-1126259446-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
#t,m,SNRL=matched_filter(strain,tl,dt)

#Second event



#Third event



#Fourth event





#plt.loglog(freqs[filtpower!=0],filtpower[filtpower!=0])


