#Jonathan Kalinowski - PHYS 512 - Asst 2

import numpy as np
from numpy.polynomial import chebyshev as cb
from matplotlib import pyplot as plt


def question1(*args,showErr=False):
    xvals=np.linspace(0.5,1,num=1000)
    y=np.log2(xvals)
    xvals = (xvals-0.75)*4
    
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
    order=0
    coeffs,err=chebfit(xvals,y,order)
    order=1
    while(abs(err)>10e-6):
        coeffs,err=chebfit(xvals,y,order)
        order=order+1
    print("Fit Order: " + str(order-1))
    print(coeffs)
    if(showErr):
        coeffs,_=chebfit(xvals,y,order+10)
        maxerr=sum(abs(coeffs[order:]))
        print("Approximate Maximum Error in Fit: "+str(maxerr))
    yvals = cb.chebval(xvals,coeffs,tensor=False  )
    xvals = xvals*0.25+0.75
    if(len(args)==1):
        a,b=np.frexp(args[0])
        
        print(cb.chebval(args[0],coeffs,tensor=False))
    plt.close()
    plt.plot(xvals,yvals,label='Cheb Fit',color='r')
    plt.scatter(xvals,y,marker='x', label='Log2(x)',s=12)
    plt.legend();plt.show()
    

def question2():
    try: 
        data=np.loadtxt('229614158_PDCSAP_SC6.txt',delimiter=',')
    except: 
        print("Error finding data file")
        return
    def calc_exp(par,x):
        #par=[c,a,d]
        c=par[0];a=par[1];d=par[2]
        y=c*np.exp(a*x)+d
        grad=np.zeros([len(x),len(par)])
        grad[:,0]=np.exp(a*x)
        grad[:,1]=c*x.transpose()*np.exp(a*x)
        grad[:,2]=np.ones(len(x))
        return y, grad
    #grab data points 3200-3230 for fit
    data=data[3200:3230]
    x=data[:,0];y=data[:,1]
    #function = ce^(-at)+d
    plt.scatter(x,y)
    guess,_=calc_exp([10000000,-.02,1],x)
    plt.plot(x,guess)