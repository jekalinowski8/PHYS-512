#Jonathan Kalinowski - PHYS 512 Asst 4
import simple_read_ligo
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as nd
import os

directory = '/home/2019/jkalin3/PHYS512/phys512_jk/asst4' #directory of data
os.chdir(directory)

def matched_filter(data,template,dt,event,plot=False,a=5):
    npoints = len(data)
    window = np.hanning(npoints) #Hanning window for data/template. Not Hamming. Hamming is pure evil. 
    fdata = np.fft.rfft(data*window) #FFT of strain
    power = fdata*np.conjugate(fdata) #Power spectrum
    ftemp = np.fft.rfft(template*window) #FFT of template
    freqs = np.fft.rfftfreq(len(data),dt) #Frequencies
    #Smooth data w/ Gaussian filter for noise model, standard deviation
    #is a=5 (I played around with a few different values)
    nn = nd.gaussian_filter1d(np.real(power),a) 
    n = np.sqrt(nn) #sqrt of noise matrix
    wfdata = fdata/n #whitened data
    wftemp = ftemp/n #whitened template
    wfdata[np.logical_or(freqs<30,freqs > 1700)] = 0 #bandpass
    wftemp[np.logical_or(freqs<30,freqs > 1700)] = 0 #redundant since it gets multiplied with the above array
    wdata = np.fft.irfft(wfdata,npoints) #whitened data in time
    wtemp = np.fft.irfft(wftemp,npoints) #whitened template in time
    fm = np.conjugate(wftemp)*wfdata #matched filter in freq
    m = np.fft.irfft(fm,npoints)/np.mean(wtemp**2) #matched filter in time
    ts = range(npoints)*dt #time values
    SNR_t = np.abs(m*np.sqrt(np.mean(wtemp**2))) #analytic snr
    SNR_o = max(m)/np.std(m) #observed SNR - matched filter divided by std
    pwtemp = np.real(wftemp*np.conjugate(wftemp)) #power spectrum of whitened template
    ptotal = np.sum(pwtemp)
    pcum = np.cumsum(pwtemp)
    fhalfpower = freqs[np.argmax(pcum>ptotal/2)]
    if plot:
        plt.figure() 
        plt.plot(ts,SNR_t)
        plt.title(event)
        plt.xlabel("$t_{offset}$ (s)")
        plt.ylabel("SNR")
    return ts,m,max(SNR_t),SNR_o,fhalfpower

#First event: GW150914
template_name='GW150914_4_template.hdf5'
th,_=simple_read_ligo.read_template(template_name)
fname='H-H1_LOSC_4_V2-1126259446-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsh1,mh1,tSNRH1,oSNRH1,fhph1 = matched_filter(strain,th,dt,"GW150914 Hanford")
fname='L-L1_LOSC_4_V2-1126259446-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsl1,ml1,tSNRL1,oSNRL1,fhpl1 = matched_filter(strain,th,dt,"GW150914 Livingston")
tcSNR1 = np.sqrt(tSNRH1**2+tSNRL1**2)
ocSNR1 = np.sqrt(oSNRH1**2+oSNRL1**2)
print("Event 1 - GW150914:")
print("Analytic SNRS:")
print("Livingston: " +  str(tSNRL1))
print("Hanford: " + str(tSNRH1))
print("Combined: "+ str(tcSNR1))
print("Observed SNRS:")
print("Livingston: " +  str(oSNRL1))
print("Hanford: " + str(oSNRH1))
print("Combined: "+ str(ocSNR1))
print("Half-Power Frequencies:")
print("Livingston: " + str(fhpl1))
print("Hanford: " + str(fhph1))
print()

#Second event: LVT151012
template_name='LVT151012_4_template.hdf5'
th,_=simple_read_ligo.read_template(template_name)
fname='H-H1_LOSC_4_V2-1128678884-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsh2,mh2,tSNRH2,oSNRH2,fhph2 = matched_filter(strain,th,dt,"LVT151012 Hanford")
fname='L-L1_LOSC_4_V2-1128678884-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsl2,ml2,tSNRL2,oSNRL2,fhpl2 = matched_filter(strain,th,dt,"LVT151012 Livingston")
tcSNR2 = np.sqrt(tSNRH2**2+tSNRL2**2)
ocSNR2 = np.sqrt(oSNRH2**2+oSNRL2**2)
print("Event 2 - LVT151012:")
print("Analytic SNRS:")
print("Livingston: " +  str(tSNRL2))
print("Hanford: " + str(tSNRH2))
print("Combined: "+ str(tcSNR2))
print("Observed SNRS:")
print("Livingston: " +  str(oSNRL2))
print("Hanford: " + str(oSNRH2))
print("Combined: "+ str(ocSNR2))
print("Half-Power Frequencies:")
print("Livingston: " + str(fhpl2))
print("Hanford: " + str(fhph2))
print()

#Third event: GW151226
template_name='GW151226_4_template.hdf5'
th,_=simple_read_ligo.read_template(template_name)
fname='H-H1_LOSC_4_V2-1135136334-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsh3,mh3,tSNRH3,oSNRH3,fhph3 = matched_filter(strain,th,dt,"GW151226 Hanford")
fname='L-L1_LOSC_4_V2-1135136334-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsl3,ml3,tSNRL3,oSNRL3,fhpl3 = matched_filter(strain,th,dt,"GW151226 Livingston")
tcSNR3 = np.sqrt(tSNRH3**2+tSNRL3**2)
ocSNR3 = np.sqrt(oSNRH3**2+oSNRL3**2)
print("Event 3 - GW151226:")  
print("Analytic SNRS:")
print("Livingston: " +  str(tSNRL3))
print("Hanford: " + str(tSNRH3))
print("Combined: "+ str(tcSNR3))
print("Observed SNRS:")
print("Livingston: " +  str(oSNRL3))
print("Hanford: " + str(oSNRH3))
print("Combined: "+ str(ocSNR3))
print("Half-Power Frequencies:")
print("Livingston: " + str(fhpl3))
print("Hanford: " + str(fhph3))
print()

#Fourth event: GW170104
template_name='GW170104_4_template.hdf5'
th,_=simple_read_ligo.read_template(template_name)
fname='H-H1_LOSC_4_V1-1167559920-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsh4,mh4,tSNRH4,oSNRH4,fhph4 = matched_filter(strain,th,dt,"GW170104 Hanford")
fname='L-L1_LOSC_4_V1-1167559920-32.hdf5'
strain,dt,utc=simple_read_ligo.read_file(fname)
tsl4,ml4,tSNRL4,oSNRL4,fhpl4 = matched_filter(strain,th,dt,"GW170104 Livingston",plot=True)
tcSNR4 = np.sqrt(tSNRH4**2+tSNRL4**2)
ocSNR4 = np.sqrt(oSNRH4**2+oSNRL4**2)
print("Event 4 - GW170104:")  
print("Analytic SNRS:")
print("Livingston: " +  str(tSNRL4))
print("Hanford: " + str(tSNRH4))
print("Combined: "+ str(tcSNR4))
print("Analytic SNRS:")
print("Livingston: " +  str(oSNRL4))
print("Hanford: " + str(oSNRH4))
print("Combined: "+ str(ocSNR4))
print("Half-Power Frequencies:")
print("Livingston: " + str(fhpl4))
print("Hanford: " + str(fhph4))
print()


