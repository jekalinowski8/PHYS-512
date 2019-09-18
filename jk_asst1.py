#Jonathan Kalinowski - PHYS 512

import numpy as np
import matplotlib.pyplot as plt


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
    hs = np.logspace(-16,1,num=numh)
    errs1 = np.abs(np.exp(x)-(-np.exp(x+2*hs)+8*np.exp(x+hs)+np.exp(x-2*hs)-8*np.exp(x-hs))/(12*hs))
    plt.plot(np.log10(hs), np.log10(errs1), label='exp(x)')
    #x=x*100
    errs2= np.abs(.01*expo(x)-((-expo(x+(2*hs))+8*expo(x+hs)+expo(x-2*hs)-8*expo(x-hs))/(12*hs)))
    plt.plot(np.log10(hs), np.log10(errs2), label='exp(0.01*x)')
    plt.legend()
    
#Question 2: 
    