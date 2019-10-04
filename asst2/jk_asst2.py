#Jonathan Kalinowski - PHYS 512 - Asst 2

import numpy as np
from numpy.polynomial import chebyshev as cb
from matplotlib import pyplot as plt

def question1(*args,plot=False,showErr=False):
    #x,y, and shifted x for chebyfit
    xvals=np.linspace(0.5,1,num=1000)
    y=np.log2(xvals)
    xvals = (xvals-0.75)*4
    #chebyshev fit
    def chebfit(x,y,order): 
        A=np.ones([len(x),order+1])
        if (order!=0):
            A[:,1]=x
        for n in range(2,order+1):
            A[:,n]=2*x*A[:,n-1]-A[:,n-2]
        A=np.matrix(A)
        d=np.matrix(y).transpose()
        lhs=np.dot(A.transpose(),A)
        rhs=np.dot(A.transpose(),d)
        coeffs=np.dot(np.linalg.inv(lhs),rhs)
        err=coeffs[-1]
        return coeffs,err
    #maximum error in truncated fit is sum of 
    #remaining fit coefficients - keep fitting until
    #we start to get coefficients below 10^-6
    order=0
    coeffs,err=chebfit(xvals,y,order)
    order=1
    while(abs(err)>10e-6):
        coeffs,err=chebfit(xvals,y,order)
        order=order+1
    yvals = cb.chebval(xvals,coeffs,tensor=False  )
    #reshift x back to original
    xvals = xvals*0.25+0.75
    #polyfit and get residuals
    pyvals = np.polyval(np.polyfit(xvals,y,6),xvals)
    rcheb=y-yvals
    rpoly=y-pyvals
    #plotting stuff    
    if plot: 
        plt.close()
        fig,(ax1,ax2)=plt.subplots(nrows=2)
        ax1.plot(xvals,pyvals,label='Poly Fit',color='k')
        ax1.plot(xvals,yvals,label='Cheb Fit',color='r')
        ax1.scatter(xvals,y,marker='x', label='Log2(x)',s=12)
        ax2.scatter(xvals,rcheb,label='Cheb Residuals',color='r')
        ax2.scatter(xvals,rpoly,label='Poly Residuals',color='k')
        ax1.legend();ax2.legend();plt.show()
    #show rms and max errors of both fits
    if(showErr):
        print('Max Cheb Err: '+str(max(abs(rcheb))))
        print('Max Poly Err: '+str(max(abs(rpoly))))
        print('RMS Cheb Err: '+str(np.std(abs(rcheb))))
        print('RMS Poly Err: '+str(np.std(abs(rpoly))))
    #calculate log2
    if(len(args)==1):
       a,b=np.frexp(args[0])
       return cb.chebval((a-0.75)*4,coeffs,tensor=False)+b

def question2():
    #load data
    try: 
        data=np.loadtxt('229614158_PDCSAP_SC6.txt',delimiter=',')
    except: 
        print("Error finding data file")
        return
    #calculate exponential and gradient
    def calc_exp(par,x):
        #fun=e^a(x-b)+c
        #par=[a,b,c]
        a=par[0,0];b=par[0,1];c=par[0,2]    
        y=np.exp(a*(x-b))+c
        grad=np.zeros([len(x),3])
        grad[:,0]=np.transpose((x-b))*np.exp(a*(x-b))
        grad[:,1]=-a*np.exp(a*(x-b))
        grad[:,2]=np.ones(len(x))
        return y, grad

    #grab data points 3200-3230 for fit
    data=data[3200:3230]
    x=data[:,0];y=data[:,1]
    #my guess for fit params
    guess=np.matrix((-25,1706.4,1))
    oguess=np.copy(guess)
    dchi2=-10000
    #newton
    while(True):
        mod,grad=calc_exp(guess,x)
        chi2=((y-mod)**2).sum()
        r=np.matrix(y-mod).transpose()
        grad=np.matrix(grad)
        lhs=grad.transpose()*grad
        rhs=grad.transpose()*r
        dg=np.linalg.inv(lhs)*rhs
        dg=np.matrix(dg).transpose()
        newguess=np.array(np.add(guess,dg))
        newmod,_=calc_exp(newguess,x)
        newchi2=((y-newmod)**2).sum()
        dchi2=newchi2-chi2
        #break if chisquare stops changing by much
        #if it gets bigger
        if(dchi2>-10e-12 and dchi2<0):
            break
        chi2=newchi2
        guess=newguess
    #plot 
    plt.close()
    plt.scatter(x,y)
    plt.plot(x,calc_exp(oguess,x)[0],label='Original Guess')
    plt.plot(x,calc_exp(guess,x)[0],label='Least Squares Fit')
    plt.legend();plt.show();
    return guess 

