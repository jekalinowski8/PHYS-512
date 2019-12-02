#Jonathan Kalinowski - 260760702 - PHYS 512 PS #5
import numpy as np
from matplotlib import pyplot as plt

n = 2000
rcyl = 250
V = 1

#TODO: Convergence conditions
def get_box(n,rcyl,V,plot=False):
    box = np.zeros((n,n))
    cylmask = np.zeros((n,n),dtype=bool)
    bmask = np.zeros((n,n),dtype=bool)
    center = n/2
    for i in range(n):
        for j in range(n):
            if (i-center)**2+(j-center)**2 <= rcyl**2:
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

    
def get_analytic_sol(n,rcyl,V):
    box = np.zeros((n,n))+V
    center = n/2
    pref = V/np.log(n/(2*rcyl))
    for i in range(n):
        for j in range(n):
            if (i-center)**2+(j-center)**2>=rcyl**2: 
                box[i,j]=V-pref*np.log((np.sqrt((i-center)**2+(j-center)**2))/rcyl)
    return box
    

def solve_relax(box,mask,plot=False):
    cbox = box.copy()
    oldbox = box.copy()
    n= len(box)
    c = int(n/2)
    V = box[c,c]
    mdif=999
    i=1
    while(mdif>V/10e6):
    #for j in range(200000):
        box[1:-1,1:-1]=(box[1:-1,0:-2]+box[1:-1,2:]+box[:-2,1:-1]+box[2:,1:-1])/4
        box[mask]=cbox[mask]
        if plot:
            plt.clf()
            plt.imshow(box)
            plt.colorbar()
            plt.pause(0.001)
        mdif = np.max(np.abs(box-oldbox))
        
        oldbox = box.copy()
        i+=1
    #print(mdif)
    print("Finished Solving by Relaxation. Total Iterations:" + str(i))
    return box
 

box, mask = get_box(n,rcyl,V)


#TODO: Masking
def solve_cgrad(box,mask,plot=False,verbose=False):
    def Ax(V,mask):
        Vuse=V.copy()
        Vuse[mask]=0
        ans=(Vuse[1:-1,:-2]+Vuse[1:-1,2:]+Vuse[2:,1:-1]+Vuse[:-2,1:-1])/4.0
        ans=ans-V[1:-1,1:-1]
        return ans
    def pad(A):
        AA=np.zeros([A.shape[0]+2,A.shape[1]+2])
        AA[1:-1,1:-1]=A
        return AA
    b=-(box[1:-1,0:-2]+box[1:-1,2:]+box[:-2,1:-1]+box[2:,1:-1])/4.0
    r=b-Ax(box,mask)
    p=r.copy()
    for k in range(10):
        Ap=(Ax(pad(p),mask))
        rtr=np.sum(r*r)
        alpha=rtr/np.sum(Ap*p)
        box=box+pad(alpha*p)
        rnew=r-alpha*Ap
        beta=np.sum(rnew*rnew)/rtr
        p=rnew+beta*p
        r=rnew
        if plot:
            plt.clf();
            plt.imshow(box)
            plt.colorbar()
            plt.pause(0.001)
        if verbose:
            print('on iteration ' + str(k))
    return box
#rho=V[1:-1,1:-1]-(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0

def solve_cgrad_hires(box,mask,rcyl,plot=False):
    n = len(box)
    if(n<500):
        box = solve_cgrad(box,mask,plot=plot)
        return box
    else:     
        #1 - solve at lower resolution (recursively)
        #2 - interpolate up
        #3 solve 
        #solve at lower resolution
        n_new = int(n/2)
        r_new = int(rcyl/2)
        V = box[n_new,n_new]
        box_new, mask_new = get_box(n_new,r_new,V)
        box_new = solve_cgrad_hires(box_new,mask_new,r_new,plot=plot)
        hires_box,hires_mask = get_box(2*n_new,2*r_new,0)
        for ix,iy in np.ndindex(box_new.shape):
            hires_box[ix,iy]=box_new[ix,iy]
            hires_box[ix+1,iy]=box_new[ix,iy]
            hires_box[ix,iy+1]=box_new[ix,iy]
            hires_box[ix+1,iy+1]=box_new[ix,iy]
        return solve_cgrad(hires_box,hires_mask,plot=plot)
        