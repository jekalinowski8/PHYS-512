#starter code
import numpy as np
import camb
from matplotlib import pyplot as plt


def get_spectrum(pars,lmax=2000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[2:1201,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt

def cal_deriv(pars,fn,lendata):
    deriv = np.zeros((len(pars),lendata))
    for i in range((len(pars))):
        h=0.001*pars[i]
        dpars1 = np.zeros(len(pars))
        dpars2=  np.zeros(len(pars))
        dpars1[i] = h
        dpars2[i] = -h
        num = fn(np.add(pars,dpars1))-fn(np.add(pars,dpars2))
        den = 2*h
        deriv[i]=num/den
    return np.array(deriv)

#plt.ion()

pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
#plt.clf()
cmb=get_spectrum(pars)
#plt.plot(cmb,zorder=10,color='r')
#plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*',color='k',zorder=5)
#plt.plot(wmap[:,0],wmap[:,1],'.',color='k',zorder=0)
chi2=sum(np.array(((cmb-wmap[:,1])/(wmap[:,2])))**2)
niter=1000
noise = np.diag(wmap[:,2])
noiseinv = np.linalg.inv(noise)
for i in range(niter):
    lam = 10
    fn = get_spectrum(pars)
    grad = cal_deriv(pars,get_spectrum,len(wmap[:,0]))
    curve = np.dot(np.dot(grad,noiseinv),grad.transpose())
    lhs = np.add(curve,lam*np.diag(curve))
    resid = wmap[:,1]-fn
    rhs = np.dot(np.dot(grad,noiseinv),resid)
    dp = np.dot(np.linalg.inv(lhs),rhs)
    pars = np.add(pars,dp)
    chi2new = sum(np.array(((get_spectrum(pars)-wmap[:,1])/(wmap[:,2])))**2)
    print(chi2new)
    if chi2new>chi2:
        lam = lam*10
    else:
        lab = lam/10
cov =np.inv(np.dot(np.dot(grad,noiseinv),grad))
    