#Jonathan Kalinowski - PHYS 512 - Asst 1 

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate 

"""
Question 1: 

a)
Let the four function points be x+h, x-h, x+2h, and x-2h
A: f'(x)= (f(x+h)-f(x-h))/2h - h^2*f^(3)(x)/6 - h^4*f^(5)(x)/120
B: f'(x)= (f(x+2h)-f(x-2h))/4h - 2h^2 f^(3)(x)/3 - 2 h^4*f^(5)(x)/15

Subtract B/4 from A to eliminate the O(h^2) error term, leaving

f'(x)=(-f(x+2h)+8f(x+h)+f(x-2h)-8f(x-h))/12h + h^4*f^(5)(x)/30


b)
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
    #Errors for exp(x) for h values
    errs1 = np.abs(np.exp(x)-(-np.exp(x+2*hs)+8*np.exp(x+hs)+np.exp(x-2*hs)-8*np.exp(x-hs))/(12*hs))
    plt.plot(np.log10(hs), np.log10(errs1), label='exp(x)')
    #Errors for exp(.01x) for h values
    errs2= np.abs(.01*expo(x)-((-expo(x+(2*hs))+8*expo(x+hs)+expo(x-2*hs)-8*expo(x-hs))/(12*hs)))
    plt.plot(np.log10(hs), np.log10(errs2), label='exp(0.01*x)')
    plt.legend()


"""
Question 2: 
Since we were given data about V(T) and dV/dT, I decided to interpolate between
two points with a cubic that matches the values and derivatives at the points.

I used this to write down some linear algebra to work out the coefficients to
solve for the cubic's coefficients between the two points. 

Since the values and derivatives of the interpolation match the data, I estimated
error with the second derivative term. 1/2f''(T)(dT^2). The second derivative
is approximated by the difference in the first derivatives of two points
divided by the spacing in between two points. dT is the spacing between the 
closest data point and the desired T value.

Running problem2(T) will produce the interpolated value and the estimated
error. problem2(T,graph=True) will show a graph of the data with the 
interpolated curve. I set the number of interpolated points to graph equal
to 10000 so that you can really see that my method works and doesn't go crazy,
but it does take a few seconds to produce the graph since my function calls
itself recursively. You can adjust the number of interpolated points are 
displayed on the graph if need be. 
"""

def problem2(T, graph=False, npts=10000): 
    #Can't extrapolate
    if (T<1.4 or T>500): 
        print("T must be between 1.4K and 500K for the interpolation")
        return
    #Load in text file
    [temps, vs, dvdts ] = np.transpose(np.loadtxt('lakeshore.txt')) 
    #Convert units to V/K
    dvdts = .001*dvdts
    #Find the two points where T lies in between
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
    #Matrix used to interpolate
    mat = [[t1*3,t1**2,t1,1],[t2**3,t2**2,t2**1,1],[3*t1**2,2*t1,1,0],[3*t2**2,2*t2,1,0]]
    matv = [[v1],[v2],[dv1],[dv2]]
    #Linear algebra :)
    coeffs = np.dot(np.linalg.inv(mat),matv)
    v= np.polyval(coeffs,T)        
    #Estimating error
    sec_deriv = (dv2-dv1)/(t2-t1)
    err = abs(0.5*sec_deriv*(T-t1)**2)
    #Producing a graph
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
    return v[0],err
                
"""
Question 3: 
    
I adapted the in-class code for the simple integrator to not repeat function
calls for the same x. I used a global dictionary variable to store function calls 
which could be easily be utilized by any of the recursive calls. I'm not 
sure if this is any more efficient and I'm sure that there's better, more
efficient ways of doing this but this seemed like a simple and effective way
to save on all repeated function evaluations. 

Here is data on the number of saved function calls by using 
some sample integrations for various functions on 0 to 1. 


                        Error Tolerance
Function: \ 0.001   \  0.000001 \ 0.000000001 \
----------------------------------------------\          
exp(x)    \    0    \     42    \     186     \
sin(x)    \    0    \     18    \     168     \
e^(-x^2)  \    0    \     42    \     342     \
-----------------------------------------------      


Run problem3(fun,a,b) and see how many function evaluations my rewrite of the
integrator saved. Error tolerance is 0.001 by default but can be adjusted as 
well. 

"""
 
def problem3(fun, a, b, tol=.001):
    #Delete 'evals' global var if it exits
    if ('evals' in globals()):
        global evals
        del evals
    
    #Simple integrator, doesn't repeat funciton evaluations
    def simple_integrate_efficient(fun, a, b, tol=.001):
        #Make a global dictionary to store funciton evaluations if it doesn't exist
        if not('evals' in globals()):
            global evals 
            evals = {}
        #Everything else is pretty much the same. Instead of evaluating the 
        #function, just look in the dictionary
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
    
    #From in-class code
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

    #Evaluate w/ and w/o repeated function calls
    ineff_f, ineff_err, ineff_neval = simple_integrate(fun,a,b,tol)
    eff_f, eff_err, eff_neval = simple_integrate_efficient(fun,a,b,tol)
    return eff_f,str(ineff_neval - eff_neval)

"""
Question 4: 
    
Running problem4() will plot the electric field from the center of a spherical
shell (R=1) as a function of the distance from the center of the sphere, 
calculated using both my integrator and scipy.integrate.quad. Both produce
the same result. There is a singularity in the integral, but both integrators 
are able to handle it without issue. 
"""

def problem4():
    #The function to integrate
    def messyboi(u,z,R):
        num = z-R*u
        denom = 1+z**2-2*z*u
        return num/(denom)**(3/2)
    #Various z values
    zs = np.linspace(0,5,num=100)
    mys = []
    quads = []
    for z in zs:
        #Do the integrals w/ my integrator and quad, plot
        my,garbage = problem3(lambda x: messyboi(x,z,1),-1,1)
        mys.append(my)
        quad, trash= integrate.quad(lambda y: messyboi(y,z,1),-1,1)
        quads.append(quad)
    plt.scatter(zs,mys,marker='o',color='k',label='My Integrator')
    plt.scatter(zs,quads,marker='x',color='r',label='Quad')
    plt.legend()
        
    
        









