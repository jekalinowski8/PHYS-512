#Jonathan Kalinowski - PHYS 512 Asst 4
import simple_read_ligo
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import ndimage as nd
plt.clf()
fname='H-H1_LOSC_4_V2-1126259446-32.hdf5'
print ('reading file ',fname)
strain,dt,utc=simple_read_ligo.read_file(fname)
#th,tl=read_template('GW150914_4_template.hdf5')
template_name='GW150914_4_template.hdf5'
th,tl=simple_read_ligo.read_template(template_name)
window = np.hamming(len(strain))
spectrum = np.fft.rfft(strain*window)
power = np.real(spectrum*np.conjugate(spectrum))
freqs = np.fft.rfftfreq(len(strain),dt)
filtpower = nd.gaussian_filter1d(power,100)
filtpower[np.logical_or(freqs<200,freqs > 1700)] = 0
N=np.fft.irfft(filtpower/power)
A=th/np.sqrt(N)
d=strain/np.sqrt(N)

m=np.fft.irfft(np.fft.rfft(A)*np.fft.rfft(d))/(A**2)


#plt.loglog(freqs[filtpower!=0],filtpower[filtpower!=0])


