#Jonathan Kalinowski - PHYS 512 - Asst 2

import numpy as np
from numpy.polynomial import chebyshev as cb
from matplotlib import pyplot as plt


def question1(*args):
    xvals=np.linspace(0.5,1,num=1000)
    y=np.log2(xvals)
    xvals = (xvals-0.75)*4
    
    def chebfit(x,y,order): 
        
        A=np.ones([len(x),order])
        A[1:]=x
        for n in range(2,order):
            A[n]=2*x*A[n-1]-A[n-2]
        ata=np.dot(A,np.transpose(A))
        coeffs=np.dot(np.dot(np.invert(ata),np.transpose(A)),y)
        return coeffs,sum(coeffs)
    
    order=0
    coeffs,err=chebfit(xvals,y,order)
    while(err>10e-6):
        order=order+1
        coeffs,err=chebfit(xvals,y,order)
        print((order,err))
        
    xvals = xvals*0.25+0.75
    yvals = cb.chebval(xvals,coeffs)
    plt.plot(xvals,yvals)
    


