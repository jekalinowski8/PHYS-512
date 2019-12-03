#Jonathan Kalinowski - 260760702 - PHYS 512 PS #5
import numpy as np
from matplotlib import pyplot as plt
import time 


#Make an nxn grounded box with a wire in the center with radius rcyl at V
#Return the box and a mask
def get_box(n,rcyl,V,bump=False,plot=False):
    box = np.zeros((n,n))
    cylmask = np.zeros((n,n),dtype=bool)
    bmask = np.zeros((n,n),dtype=bool)
    center = n/2
    for i in range(n):
        for j in range(n):
            if (i-center)**2+(j-center)**2 <= rcyl**2:
                box[i,j]=V
                cylmask[i,j]=True
            elif bump: 
                 if (i-center)**2+(j-center-rcyl)**2 <= (.2*rcyl)**2:
                     box[i,j]=V
                     cylmask[i,j]=True
    bmask = np.zeros((n,n),dtype=bool)
    bmask[:,0]=True
    bmask[:,-1]=True
    bmask[0,:]=True
    bmask[-1,:]=True
    mask = cylmask + bmask
    if plot:
        plt.clf()
        plt.imshow(box) 
    return box,mask

#Get an 'analytic' solution of the potential for a wire w/o a box
def get_analytic_sol(n,rcyl,V):
    box = np.zeros((n,n))+V
    center = n/2
    pref = V/np.log(n/(2*rcyl))
    for i in range(n):
        for j in range(n):
            if (i-center)**2+(j-center)**2>=rcyl**2: 
                box[i,j]=V-pref*np.log((np.sqrt((i-center)**2+(j-center)**2))/rcyl)
    return box
    
#Solve using relaxation
def solve_relax(box,mask,verbose=False,plot=False):
    t1 = time.time()
    cbox = box.copy()
    oldbox = box.copy()
    mdif=999
    i=1
    #Converges when the maximum change in iterations is less than 10e-6
    while(mdif>1e-6):
        #Discrete Poisson eq. 
        box[1:-1,1:-1]=(box[1:-1,0:-2]+box[1:-1,2:]+box[:-2,1:-1]+box[2:,1:-1])/4
        #Re-apply the mask
        box[mask]=cbox[mask]
        if plot:
            plt.clf()
            plt.imshow(box)
            plt.colorbar()
            plt.pause(0.001)
        mdif = np.max(np.abs(box-oldbox))
        oldbox = box.copy()
        i+=1
        if(verbose):
            print(mdif)
    print("Finished Solving by Relaxation. Total Iterations:" + str(i) + " Total Time Elapsed: " + str(time.time()-t1) + " seconds.")
    return box
 
#Solve using conjugate gradient - code adapted from in-class code (https://github.com/sievers/phys512/blob/master/conjugate_gradient/laplace_conjgrad.py)
def solve_cgrad(box,mask,verbose=False,plot=False):
    t1 = time.time()
    #Compute Ax, the matrix representation of the Laplacian
    def Ax(V,mask):
        Vuse=V.copy()
        Vuse[mask]=0
        ans=(Vuse[1:-1,:-2]+Vuse[1:-1,2:]+Vuse[2:,1:-1]+Vuse[:-2,1:-1])/4.0
        ans=ans-V[1:-1,1:-1]
        return ans
    #Pad matrix with 0s on boundary, which get cut off by Ax()
    def pad(A):
        AA=np.zeros([A.shape[0]+2,A.shape[1]+2])
        AA[1:-1,1:-1]=A
        return AA
    #Conjugate gradient steps
    b=-(box[1:-1,0:-2]+box[1:-1,2:]+box[:-2,1:-1]+box[2:,1:-1])/4.0
    box = 0*box
    r=b-Ax(box,mask)
    p=r.copy()
    mdif = 999
    i = 1
    while(mdif>1e-6):
        oldbox = box.copy()
        Ap=(Ax(pad(p),mask))
        rtr=np.sum(r*r)
        alpha=rtr/np.sum(Ap*p)
        box=box+pad(alpha*p)
        rnew=r-alpha*Ap
        beta=np.sum(rnew*rnew)/rtr
        p=rnew+beta*p
        r=rnew
        mdif = np.max(np.abs(box-oldbox))
        if plot:
            plt.clf();
            plt.imshow(box)
            plt.colorbar()
            #plt.clim(vmin=0,vmax=1)
            plt.pause(0.001)
        if verbose:
            print(mdif)
        i+=1
    print("Finished Solving by Relaxation. Total Iterations:" + str(i) + " Total Time Elapsed: " + str(time.time()-t1) + " seconds.")
    return box

#Recursively solve a high-resolution box w/ conjugate gradient by solving first at lower resolutions
#and interpolating up. Does not work. 
def solve_cgrad_hires(box,mask,rcyl,verbose=False,plot=False,level=0):
    #Solve box less than n=64 directly, it's fast
    n = len(box)
    obox = box.copy()
    print("n="+str(n))
    V = box[n//2,n//2]
    if(n<64):
        box = solve_cgrad(box,mask,verbose=verbose,plot=plot)
        return box
    else:     
        #1 - solve at lower resolution (recursively)
        #2 - interpolate up
        #3 - solve higher resolution with lower res solution
        n_new = int(n/2)
        r_new = int(rcyl/2)
        V = box[n_new,n_new]
        box_new, mask_new = get_box(n_new,r_new,V)
        box_new = solve_cgrad_hires(box_new,mask_new,r_new,verbose=verbose,plot=plot,level=level+1)
        sbox = np.zeros((n,n))
        for ix,iy in np.ndindex(box_new.shape):
            sbox[2*ix,2*iy]=box_new[ix,iy]
            sbox[2*ix+1,2*iy]=box_new[ix,iy]
            sbox[2*ix,2*iy+1]=box_new[ix,iy]
            sbox[2*ix+1,2*iy+1]=box_new[ix,iy]
        sbox[mask]=obox[mask]
        plt.clf();plt.imshow(sbox);plt.colorbar();plt.pause(10)
        return solve_cgrad(sbox,mask,verbose=verbose,plot=plot)
    
#Solve temperature vs time for a box, initially at T=0, with one size 
#having its temperature increased linearly with time. 
def solve_heat(n,rheat=1,verbose=False,plot=False):
    box = np.zeros((n,n))
    i=1
    #D=.04
    while(i<1000):
        box[1:-1,1:-1]=(box[1:-1,0:-2]+box[1:-1,2:]+box[:-2,1:-1]+box[2:,1:-1])/4
        #box[1:-1,1:-1]=(box[1:-1,0:-2]+box[1:-1,2:]+box[:-2,1:-1]+box[2:,1:-1])/4-D*box[1:-1,1:-1]
        if i<10:
            box[:,0] = rheat*i
        if plot:
            plt.clf()
            plt.imshow(box)
            plt.colorbar()
            plt.pause(0.001)
        i+=1
        print(i)
        if i%100 == 0:
            plt.plot(box[n//2,:],label= "t="+str(i))
    plt.xlabel("x")
    plt.ylabel("T(x,t)")
    plt.legend()
    