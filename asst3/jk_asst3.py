#starter code
import numpy as np
import camb
from matplotlib import pyplot as plt

plt.close()
def get_spectrum(pars,lmax=2000,tau_fixed=False):
    if (tau_fixed):
        H0,ombh2,omch2,As,ns=pars
        tau=0.05
    else:
        H0,ombh2,omch2,tau,As,ns=pars
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[2:1201,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt

def cal_deriv(pars,fn,lendata,tau_fixed):
    deriv = np.zeros((len(pars),lendata))
    for i in range((len(pars))):
        h=0.01*pars[i]
        dpars1 = np.zeros(len(pars))
        dpars2=  np.zeros(len(pars))
        dpars1[i] = h
        dpars2[i] = -h
        num = fn(np.add(pars,dpars1),tau_fixed=tau_fixed)-fn(np.add(pars,dpars2),tau_fixed=tau_fixed)
        den = 2*h
        deriv[i]=num/den
    return np.array(deriv)

#plt.ion()

pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
cmb=get_spectrum(pars)
#plt.plot(cmb,zorder=10,color='r')
#plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*',color='k',zorder=5)
#plt.plot(wmap[:,0],wmap[:,1],'.',color='k',zorder=0)
#chi2=sum(np.array(((cmb-wmap[:,1])/(wmap[:,2])))**2)


niter=10
lam=1
noise = np.diag(wmap[:,2])
noiseinv = np.linalg.inv(noise**2)
fix_od=False
do_newton = False
if not 'cov' in locals() and do_newton: 
    if (fix_od):
         pars = [pars[0],pars[1],pars[2],pars[4],pars[5]]
         cmb = get_spectrum(pars,tau_fixed=True)
         dchi2 = 999
         while (dchi2>0.000001):
            grad = cal_deriv(pars,get_spectrum,len(wmap[:,0]),True)
            curve = np.dot(np.dot(grad,noiseinv),grad.transpose())
            lhs = np.add(curve,lam*np.diag(curve))
            resid = wmap[:,1]-cmb
            rhs = np.dot(np.dot(grad,noiseinv),resid)
            dp = np.dot(np.linalg.inv(lhs),rhs)
            chi2=sum(np.array(((cmb-wmap[:,1])/(wmap[:,2])))**2)
            pars = np.add(pars,dp)
            newcmb = get_spectrum(pars,tau_fixed=True)
            chi2new = sum(np.array(((newcmb-wmap[:,1])/(wmap[:,2])))**2)
            print(chi2new)
            if chi2new>chi2:
                lam = lam*10
            else:
                lam = lam/10
            dchi2=abs((chi2new-chi2)/chi2new)
            chi2=chi2new
            cmb=newcmb
                
    else:
        cmb = get_spectrum(pars)
        dchi2 = 999
        while (dchi2>0.000001):
            grad = cal_deriv(pars,get_spectrum,len(wmap[:,0]),False)
            curve = np.dot(np.dot(grad,noiseinv),grad.transpose())
            lhs = np.add(curve,lam*np.diag(curve))
            resid = wmap[:,1]-cmb
            rhs = np.dot(np.dot(grad,noiseinv),resid)
            dp = np.dot(np.linalg.inv(lhs),rhs)
            chi2=sum(np.array(((cmb-wmap[:,1])/(wmap[:,2])))**2)
            pars = np.add(pars,dp)
            newcmb = get_spectrum(pars)
            chi2new = sum(np.array(((newcmb-wmap[:,1])/(wmap[:,2])))**2)
            print(chi2new)
            if chi2new>chi2:
                lam = lam*10
            else:
                lam = lam/10
            dchi2 = abs((chi2new-chi2)/chi2new)
            chi2=chi2new
            cmb = newcmb
            
print("Finished Gauss-Newton. Params are "+str(pars))
cmb = get_spectrum(pars,tau_fixed=fix_od)
chi2=sum(np.array(((cmb-wmap[:,1])/(wmap[:,2])))**2)
cov =np.dot(np.dot(grad,noiseinv),grad.transpose())

def take_step():
    return np.asarray([10,0.01,0.1,0.01,1e-9,.01])*np.random.randn(6)
def take_step_cov(covmat):
    mychol=np.linalg.cholesky(covmat)
    return np.dot(mychol,np.random.randn(covmat.shape[0]))

#don't evaulate camb if tau goes negative
nstep=1000
npar=len(pars)
chains_new=np.zeros([nstep,npar])
scale_fac=1
chisqvec_new=np.zeros(nstep)
num_accept=0
for i in range(nstep):
    if (fix_od):
        break
    new_pars=pars+take_step()*scale_fac
    if (new_pars[3]<=0):
        i=i-1
        print("optical depth negative")
        continue
    try: 
        new_cmb=get_spectrum(new_pars)
    except: 
        print("camb err")
        i=i-1
        continue
    new_chi2=np.sum( (wmap[:,1]-new_cmb)**2/wmap[:,2]**2)
    delta_chisq=new_chi2-chi2
    prob=np.exp(-0.5*delta_chisq)
    accept=np.random.rand(1)<prob
    if accept:
        print("step accepted")
        num_accept=num_accept+1
        pars=new_pars
        cmb=new_cmb
        chi2=new_chi2
    chains_new[i,:]=pars
    chisqvec_new[i]=chi2
    print("Accepted ratio: "+str(num_accept/(i+1)))
    
fit_params=np.mean(chains_new,axis=0)




plt.ion()

