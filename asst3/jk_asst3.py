#starter code
import numpy as np
import camb
from matplotlib import pyplot as plt
import time as t

plt.close()

#calculate the power spectrum
def get_spectrum(pars,lmax=2000,tau_fixed=False):
    if (tau_fixed):
        H0,ombh2,omch2,As,ns=pars
        tau=0.0544
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

#calculate the derivative of the power spectrum
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
    return np.matrix(deriv).transpose()

#default parameters & data
pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
cmb=get_spectrum(pars)

#plot and calculate chi2:
#plt.plot(cmb,zorder=10,color='r')
#plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*',color='k',zorder=5)
#plt.plot(wmap[:,0],wmap[:,1],'.',color='k',zorder=0)
#chi2=sum(np.array(((cmb-wmap[:,1])/(wmap[:,2])))**2)

#Newton's method w/ Levenberg-Marquardt
niter=10
lam=1
noiseinv = np.linalg.inv(np.diag(wmap[:,2])**2)
fix_od=False
do_newton = False

if do_newton:
    print('Running Newton w/ Levenberg–Marquardt. Chi2 values are:')
    if (fix_od):
         pars = [pars[0],pars[1],pars[2],pars[4],pars[5]]
         cmb = get_spectrum(pars,tau_fixed=True)
         dchi2 = 999
         while (dchi2>0.000001):
            A = cal_deriv(pars,get_spectrum,len(wmap[:,0]),True)
            cov = np.dot(np.dot(A.transpose(),noiseinv),A)
            lhs = np.add(cov,lam*np.diag(cov))
            resid = np.matrix((wmap[:,1]-cmb))
            resid = resid.reshape(-1, 1)
            rhs = np.dot(np.dot(A.transpose(),noiseinv),resid)
            dp = np.dot(np.linalg.inv(lhs),rhs)
            chi2=sum(np.array(((cmb-wmap[:,1])/(wmap[:,2])))**2)
            pars = np.add(pars,dp.transpose())
            pars = np.ravel(np.asarray(pars.ravel()))
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
            A = cal_deriv(pars,get_spectrum,len(wmap[:,0]),False)
            cov = np.dot(np.dot(A.transpose(),noiseinv),A)
            lhs = np.add(cov,lam*np.diag(cov))
            resid = wmap[:,1]-cmb
            resid = resid.reshape(-1, 1)
            rhs = np.dot(np.dot(A.transpose(),noiseinv),resid)
            dp = np.array(np.dot(np.linalg.inv(lhs),rhs))
            chi2=sum(np.array(((cmb-wmap[:,1])/(wmap[:,2])))**2)
            pars = np.add(pars,dp.transpose())       
            new_chi2=np.sum( (wmap[:,1]-new_cmb)**2/wmap[:,2]**2)
            pars = pars.ravel()
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
    newton_pars = pars
           
print("Finished Gauss-Newton. Params are "+str(pars))
cmb = get_spectrum(pars,tau_fixed=fix_od)
chi2=sum(np.array(((cmb-wmap[:,1])/(wmap[:,2])))**2)
try:
    cov = np.linalg.inv(np.dot(np.dot(A.transpose(),noiseinv),A))
    print('Parameter Errors are ' + str(np.sqrt(np.diag((cov)))))
except:
    pass


#MCMC
   
def take_step_cov(covmat):
    mychol=np.linalg.cholesky(covmat)
    return np.ravel(np.dot(mychol,np.random.randn(covmat.shape[0])))
def take_step():
    param_scalings=np.asarray([65,0.02,0.1,2e-9,0.96])
    return param_scalings*np.random.randn(len(param_scalings))
now = t.time()
nstep=10000
npar=len(pars)
chains=np.zeros([nstep,npar+1])
#scale_fac=.4 #best scale factor floated tau
scale_fac = .6 #best scale factor fixed tau
num_accept=0
chains_old = np.loadtxt('chains_old.csv',delimiter=',')
chains_old = chains_old[1000:9000,0:6]
cov = np.cov(chains_old[int(nstep/2):9000,:].T)
pars = np.mean(chains_old,axis=0)
fix_od=True
i=0
if(not fix_od):
    print("Starting MCMC: Tau Floating")
    while(i<nstep):
        new_pars=pars+take_step_cov(cov)*scale_fac
        if (new_pars[3]<=0):
            continue
        try:        
            new_cmb=get_spectrum(new_pars)
        except CAMBError:
            continue
        new_chi2=np.sum( (wmap[:,1]-new_cmb)**2/wmap[:,2]**2)
        delta_chisq=new_chi2-chi2
        prob=np.exp(-0.5*delta_chisq)
        accept=np.random.rand(1)<probtake_step_cov(cov)
        if accept:
            num_accept=num_accept+1
            pars=new_pars
            cmb=new_cmb
            chi2=new_chi2
        chains[i,:]=np.append(pars,chi2)
        if (i%5==0 and i!=0):
            elapsed = t.time()
            print("Iteration Number: " + str(i)+ " ; Elapsed Time: " + str(elapsed-now) + " s")
            now = t.time()
            np.savetxt("chains8.csv",chains,delimiter=',')
            print("Accepted ratio: "+str(num_accept/(i+1)))
        i=i+1
else:
    scale_fac = .008
    print("Starting MCMC: Tau Fixed")
    cov = np.delete(cov,3,axis=0)
    cov = np.delete(cov,3,axis=1)
    pars = np.asarray([74,0.0231,0.106,2e-9,0.99])
    chains=np.zeros([nstep,npar])
    while(i<nstep):
        #new_pars=pars+take_step_cov(cov)*scale_fac
        new_pars=pars+take_step()*scale_fac
        try:
            new_cmb=get_spectrum(new_pars,tau_fixed=True)
        except CAMBError:
            continue
        new_chi2=np.sum( (wmap[:,1]-new_cmb)**2/wmap[:,2]**2)
        delta_chisq=new_chi2-chi2
        prob=np.exp(-0.5*delta_chisq)
        accept=np.random.rand(1)<prob
        if accept:
            num_accept=num_accept+1
            pars=new_pars
            cmb=new_cmb
            chi2=new_chi2
        chains[i,:]=np.append(pars,chi2)
        if (i%5==0 and i!=0):
            elapsed = t.time()
            print("Iteration Number: " + str(i)+ " ; Elapsed Time: " + str(elapsed-now) + " s")
            now = t.time()
            np.savetxt("chains1_tau.csv",chains,delimiter=',')
            print("Accepted ratio: "+str(num_accept/(i+1)))
        i=i+1
   
   
   
