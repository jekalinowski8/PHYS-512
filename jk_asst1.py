#Jonathan Kalinowski - PHYS 512

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#Question 1: 
#a)
"""
Let the four function points be x+h, x-h, x+2h, and x-2h
A: f'(x)= (f(x+h)-f(x-h))/2h - h^2*f^(3)(x)/6 - h^4*f^(5)(x)/120
B: f'(x)= (f(x+2h)-f(x-2h))/4h - 2h^2 f^(3)(x)/3 - 2 h^4*f^(5)(x)/15

Subtract B/4 from A to eliminate the O(h^2) error term, leaving


f'(x)=(-f(x+2h)+8f(x+h)+f(x-2h)-8f(x-h))/12h + h^4*f^(5)(x)/30
"""

#b)
"""
With the consideration of machine precsion, the error in f'(x) becomes: 

Error = h^4*f^(5)(x)/30 + e*f(x)/12h^2

where e is the machine epsilon, the floating point error. Taking a derivative 
w.r.t h and setting it equal to 0, and then solving for h, gives an approximate
value of h that minimizes the error of the derivative estimate. This is:

h=(5e*f(x)/8*f^(5)(x))^(1/5) ~ e^(1/5)


So we expect that for exp(x), where the ratio of the function evaluation and 
its fifth derivative to be 1, that the value of h that produces the smallest 
error should be about e^(1/5). With e taking on a value of about 10^-16, 
this gives an h of about 6*10^-4 which is about 10^-3.2

For exp(0.01x), (0.01)^5 pops out from the fifth derivative, meaning that the 
optimal h is about 6*10^-4/.01=10^-1.2

Running problem1() will show a log-log plot for error as a function of h
for exp(x) and exp(0.01x). 
The plot shows that the two calculations of the optimal value of h are
indeed (roughly) correct!
"""

def problem1(numh = 1000, x=1): 
    #Helper function 
    def expo(x): 
        return np.exp(x*.01)
    plt.close()
    #Take values of h from 10^-16 to 1
    hs = np.logspace(-16,1,num=numh)
    errs1 = np.abs(np.exp(x)-(-np.exp(x+2*hs)+8*np.exp(x+hs)+np.exp(x-2*hs)-8*np.exp(x-hs))/(12*hs))
    plt.plot(np.log10(hs), np.log10(errs1), label='exp(x)')
    errs2= np.abs(.01*expo(x)-((-expo(x+(2*hs))+8*expo(x+hs)+expo(x-2*hs)-8*expo(x-hs))/(12*hs)))
    plt.plot(np.log10(hs), np.log10(errs2), label='exp(0.01*x)')
    plt.legend()

#Question 2: 
"""


"""
def problem2(T, graph=False): 
    if (T<1.4 or T>500): 
        print("T must be between 1.4K and 500K for the interpolation")
        return
    [temps, vs, dvdts ] = np.transpose(np.loadtxt('lakeshore.txt')) 
    dvdts = .001*dvdts
    i=0
    while(temps[i]<T):
        i=i+1
    i=i-1
    t1 = temps[i]
    t2 = temps[i+1]
    v1 = vs[i]
    v2 = vs[i+1]
    dv1 = dvdts[i]
    dv2 = dvdts[i+1]
    mat = [[t1*3,t1**2,t1,1],[t2**3,t2**2,t2**1,1],[3*t1**2,2*t1,1,0],[3*t2**2,2*t2,1,0]]
    matv = [[v1],[v2],[dv1],[dv2]]
    coeffs = np.dot(np.linalg.inv(mat),matv)
    v= np.polyval(coeffs,T)        
    sec_deriv = (dv2-dv1)/(t2-t1)
    err = abs(0.5*sec_deriv*(T-t1)**2)
    if(graph):
        plt.close()
        ts = np.linspace(1.40,499,num=10000)
        np.append(ts,temps)
        interp = []
        errs = []
        for t in ts: 
            val, err = problem2(t)
            interp.append(val)
            errs.append(err)
        plt.scatter(temps, vs, color='b', label='Data')
        plt.scatter(T,v,color='r',marker='x', label='Input')
        plt.plot(ts,interp, color='k', label='Interpolated Curve')
        plt.legend()
    if (not graph):
        print("(Voltage, Error Estimate)=")
    return v[0],err
                
    

    
    
def problem3(fun, a, b, tol=.001):
    if ('evals' in globals()):
        global evals
        del evals
    def simple_integrate_efficient(fun, a, b, tol=.001):
        if not('evals' in globals()):
            global evals 
            evals = {}
        x=np.linspace(a,b,5) 
        for xs in x: 
            if xs not in evals: 
                evals[xs]=fun(xs)
        f1=(evals[x[0]]+4*evals[x[2]]+evals[x[4]])/6.0*(b-a)
        f2=(evals[x[0]]+4*evals[x[1]]+2*evals[x[2]]+4*evals[x[3]]+evals[x[4]])/12.0*(b-a)
        myerr=np.abs(f2-f1)
        if (myerr<tol):
            return (16.0*f2-f1)/15.0,myerr,len(evals)
        else:   
            mid=0.5*(b+a)
            f_left,err_left,trash=simple_integrate_efficient(fun,a,mid,tol/2.0)
            f_right,err_right,trash2=simple_integrate_efficient(fun,mid,b,tol/2.0)
            f=f_left+f_right
            err=err_left+err_right
            return f,err,len(evals)
    
    
    def simple_integrate(fun,a,b,tol):
        x=np.linspace(a,b,5)
        y=fun(x)
        neval=len(x) 
        f1=(y[0]+4*y[2]+y[4])/6.0*(b-a)
        f2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12.0*(b-a)
        myerr=np.abs(f2-f1)
        if (myerr<tol):
            return (16.0*f2-f1)/15.0,myerr,neval
        else:
            mid=0.5*(b+a)
            f_left,err_left,neval_left=simple_integrate(fun,a,mid,tol/2.0)
            f_right,err_right,neval_right=simple_integrate(fun,mid,b,tol/2.0)
            neval=neval+neval_left+neval_right
            f=f_left+f_right
            err=err_left+err_right
            return f,err,neval

    ineff_f, ineff_err, ineff_neval = simple_integrate(fun,a,b,tol)
    eff_f, eff_err, eff_neval = simple_integrate_efficient(fun,a,b,tol)

    return eff_f,"Saved Function Calls: " +str(ineff_neval - eff_neval)




def problem4():
    def messyboi(u,z,R):
        num = z-R*u
        denom = 1+z**2-2*z*u
        return num/(denom)**(3/2)
    zs = np.linspace(0,5,num=100)
    for z in zs: 
        my,garbage = problem3(lambda x: messyboi(x,z,1),-1,1)
        plt.scatter(z,my,marker='o',color='k',label='My Integrator')
        quad, trash= sp.integrate.quad(lambda y: messyboi(y,z,1),-1,1)
        plt.scatter(z,quad,marker='x',color='r',label='Quad')
        
    
        









