#Jonathan Kalinowski - PHYS 512 Asst 4
import simple_read_ligo
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as nd
import os

directory = '/home/2019/jkalin3/PHYS512/phys512_jk/asst4' #directory of data
os.chdir(directory)

def matched_filter(data,template,dt,plot=False,a=5):
    npoints = len(data)
    window = np.hamming(npoints) #Hamming window for data/template
    fdata = np.fft.rfft(data*window) #FFT of strain
    power = fdata*np.conjugate(fdata) #Power spectrum
    ftemp = np.fft.rfft(template*window) #FFT of template
    freqs = np.fft.rfftfreq(len(data),dt) #Frequencies
    #Smooth data w/ Gaussian filter for noise model, standard deviation
    #is a=5 (I played around with a few different values)
    nn = nd.gaussian_filter1d(np.real(power),a) 
    n = np.sqrt(nn) #sqrt of noise matrix
    wfdata = fdata/n #whitened data
    wfdata[np.logical_or(freqs<300,freqs > 1700)] = 0 #bandpass
    wftemp = ftemp/n #whitened template
    fm = np.conjugate(wftemp)*wfdata #matched filter in freq
    m = np.fft.irfft(fm,len(fm)) #matched filter in time
    ts = range(len(fdata))*(dt*2) #time values
    wdata = np.fft.irfft(wfdata,npoints) #whitened data in time
    wtemp = np.fft.irfft(wftemp,npoints) #whitened template in time
    #wtemp = wtemp/np.std(wtemp) #normalize the whitened template? 
    SNR_t = np.abs(m/np.sqrt(np.mean(wtemp**2)))
    #m = np.abs(m/np.std(m))
    SNR_o = max(m)/np.std(m) #observed SNR
    if plot:
        plt.figure(np.random.randint(0,100))
        #plt.loglog(fdata)
        #plt.loglog(n)
        #plt.plot(wdata)
        #plt.plot(ts,wtemp)
        #plt.plot(ts,SNR)
        plt.plot(ts,m)
        plt.xlabel("t (s)")
        plt.ylabel("m(t)")
    return ts,m,SNR_t,SNR_o

#First event
template_name='GW150914_4_template.hdf5'
th,tl=simple_read_ligo.read_template(template_name)
fname='H-H1_LOSC_4_V2-1126259446-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsh1,mh1,_,SNRH1 = matched_filter(strain,th,dt,plot=True)
fname='L-L1_LOSC_4_V2-1126259446-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsl1,ml1,_,SNRL1 = matched_filter(strain,tl,dt,plot=True)
CSNR1 = np.sqrt(SNRH1**2+SNRL1**2)
print("Event 1 SNRS:")
print("Livingston: " +  str(SNRL1))
print("Hanford: " + str(SNRH1))
print("Combined: "+ str(CSNR1))

#Second event
template_name='LVT151012_4_template.hdf5'
th,tl=simple_read_ligo.read_template(template_name)
fname='H-H1_LOSC_4_V2-1128678884-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsh2,mh2,_,SNRH2 = matched_filter(strain,th,dt,plot=True)
fname='L-L1_LOSC_4_V2-1128678884-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsl2,ml2,_,SNRL2 = matched_filter(strain,tl,dt,plot=True)
CSNR2 = np.sqrt(SNRH2**2+SNRL2**2)
print("Event 2 SNRS:")
print("Livingston: " +  str(SNRL2))
print("Hanford: " + str(SNRH2))
print("Combined: "+ str(CSNR2))



#Third event
template_name='GW151226_4_template.hdf5'
th,tl=simple_read_ligo.read_template(template_name)
fname='H-H1_LOSC_4_V2-1135136334-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsh3,mh3,_,SNRH3 = matched_filter(strain,th,dt,plot=True)
fname='L-L1_LOSC_4_V2-1135136334-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsl3,ml3,_,SNRL3 = matched_filter(strain,tl,dt,plot=True)
CSNR3 = np.sqrt(SNRH3**2+SNRL3**2)
print("Event 3 SNRS:")
print("Livingston: " +  str(SNRL3))
print("Hanford: " + str(SNRH3))
print("Combined: "+ str(CSNR3))



#Fourth event
template_name='GW170104_4_template.hdf5'
th,tl=simple_read_ligo.read_template(template_name)
fname='H-H1_LOSC_4_V1-1167559920-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsh4,mh4,_,SNRH4 = matched_filter(strain,th,dt,plot=True)
fname='L-L1_LOSC_4_V1-1167559920-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsl4,ml4,_,SNRL4 = matched_filter(strain,tl,dt,plot=True)
CSNR4 = np.sqrt(SNRH4**2+SNRL4**2)
print("Event 4 SNRS:")
print("Livingston: " +  str(SNRL4))
print("Hanford: " + str(SNRH4))
print("Combined: "+ str(CSNR4))






