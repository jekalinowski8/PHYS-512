import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
#TODO: 3D
#TODO: Boundary conditions
#TODO: Optimize
#TODO: Questions

G=1 #gravitationnal constant
ic = 'collapse' #initial conditions

if ic == 'stationary':
    pos = np.array([[0.5,0.5]])
    vel = np.array([[0,0]])
    m = np.array([1])
elif ic == 'orbit':
    pos = np.array(([[0.5,0.5],[0.75,0.5]]))
    vel = np.array(([[0,0],[0,63.24]])) #63.24
    m = np.array([1000,1])
elif ic == 'collapse':
    pos = np.array(([[0.5,0.5],[0.75,0.5]]))
    vel = np.array(([[0,0],[0,20]]))
    m = np.array([100,.01])
else:
    nparticles=1000
    pos = np.random.rand(nparticles,2)
    vel = (np.random.rand(nparticles,2)*2-1)*0
    m = np.random.rand(nparticles)
    
def cal_force(x,m,printE=False): #calculate force between 2 particles
    def gravity(p1,p2,m1,m2):
        # gravitational force of bul2 on bul1 
        epsilon = 0.05
        x1=p1[0];x2=p2[0];y1=p1[1];y2=p2[1]
        num = G*m1*m2
        den = ((x1-x2)**2+(y1-y2)**2+epsilon**2)**(3/2)
        return np.array([-(num/den)*(x1-x2),-(num/den)*(y1-y2)])
    nparticles = len(x)
    forces=np.zeros((nparticles,2))
    for i in range(nparticles):
        for j in range(nparticles):
            if (i!=j):
                forces[i]=gravity(x[i],x[j],m[i],m[j]) #here just the gravity
    return forces

def cal_force_mesh(r,m,grid=100,eps=0.05,periodic=False):  
    x = r[:,0]; y = r[:,1]
    if not periodic: 
        grid = 2*grid
        density,bx,by= np.histogram2d(x,y,bins=grid,range=[[0, 2], [0, 2]],weights=m)
        density[grid:,:]=0
        density[:,grid:]=0
    else: 
        density,bx,by= np.histogram2d(x,y,bins=grid,range=[[0, 1], [0, 1]],weights=m)
    density = density.transpose()/(grid**2)
    #plt.figure();plt.imshow(density)
    frho = np.fft.fft2(density)
    ks = np.fft.fftfreq(grid,d=1/grid)
    k2 = np.zeros((grid,grid))
    k2 = np.tile(ks**2,(grid,1))
    k2=k2+k2.transpose()
    k2[0,0]=np.Inf
    fphi = -4*np.pi*G*frho/k2
    phi = np.fft.irfft2(fphi)
    #phi =np.real(np.fft.ifft2(fphi))
    #if not periodic:
    #    grid = grid//2
    #    phi = phi[:grid,:grid]
    #plt.figure();plt.imshow(phi);assert(1==0)
    xgrad,ygrad = np.gradient(-1*phi,1/grid)
    forces = np.empty((len(x),2))
    bx = bx[:grid]
    digx = np.digitize(x,bx)-1
    digy = np.digitize(y,bx)-1
    a = np.logical_and(digx<grid,digy<grid)
    digx = digx[a]; digy = digy[a]
    forces = np.array([xgrad[digx,digy],ygrad[digx,digy]])
    return forces.transpose()
 
cal_fn = cal_force_mesh
def take_step_RK4(dt,pos,v,m,periodic=False):
    n = np.shape(m)[0]
    '''Integration of d2y/dt2=f, here f is the acceleration'''
    x=pos[:,0];y=pos[:,1];vx=v[:,0];vy=v[:,1];
    pos1=np.empty((n,2))
    pos1[:,0]=x
    pos1[:,1]=y
    f1=cal_fn(pos1, m)/np.array([m,m]).transpose()

    x2=x+(dt/2)*vx
    y2=y+(dt/2)*vy
    pos2=np.empty((n,2))
    pos2[:,0]=x2
    pos2[:,1]=y2
    f2=cal_fn(pos2,m)/np.array([m,m]).transpose()
    x3=x+(dt/2)*vx+(dt*dt/4)*f1[:,0]
    y3=y+(dt/2)*vy+(dt*dt/4)*f1[:,1]
    pos3=np.empty((n,2))  
    pos3[:,0]=x3
    pos3[:,1]=y3
    f3=cal_fn(pos3,m)/np.array([m,m]).transpose()
    x4=x+(dt)*vx+(dt*dt/2)*f2[:,0]
    y4=y+(dt)*vy+(dt*dt/2)*f2[:,1]
    pos4=np.empty((len(x),2))
    pos4[:,0]=x4
    pos4[:,1]=y4
    f4=cal_fn(pos4,m)/np.array([m,m]).transpose()
    vxfinal=vx+(dt/6)*(f1+2*f2+2*f3+f4)[:,0]
    vyfinal=vy+(dt/6)*(f1+2*f2+2*f3+f4)[:,1]
    xfinal=x+dt*vx+((dt*dt/6)*(f1+f2+f3))[:,0]
    yfinal=y+dt*vy+((dt*dt/6)*(f1+f2+f3))[:,1]
    posfinal=np.empty((len(x),2))
    vfinal=np.empty((len(x),2))
    posfinal[:,0]=xfinal
    posfinal[:,1]=yfinal
    vfinal[:,0]=vxfinal
    vfinal[:,1]=vyfinal
    return posfinal,vfinal
    #return pos,v

"""
"""
def take_step_euler(dt,pos,v,m,printE=False,period=False):
    n = np.shape(m)[0]
    x=pos[:,0];y=pos[:,1];vx=v[:,0];vy=v[:,1];
    if printE:
        f4,V=cal_force_mesh(pos,v,m,printE=True)
        f4=f4/np.array([m,m]).transpose()
    else:
        f4=cal_force_mesh(pos,v,m)/np.array([m,m]).transpose()
    vxfinal=vx+dt*f4[:,0]
    vyfinal=vy+dt*f4[:,1]
    xfinal=x+dt*vxfinal
    yfinal=y+dt*vyfinal
    posfinal=np.empty((n,2))
    vfinal=np.empty((n,2))
    posfinal[:,0]=xfinal
    posfinal[:,1]=yfinal
    vfinal[:,0]=vxfinal
    vfinal[:,1]=vyfinalf
    return posfinal,vfinal

def calc_E(r,m,v):
    T = np.sum(0.5*m*(v[:,0]**2+v[:,1]**2))
    V=0
    for i in range(len(m)):
        for j in range(len(m)):
            if i != j:
                print(1)
    
if __name__ == '__main__':
    T=1
    traj = np.array([])
    t=0
    dt=0.0001
    plt.clf()
    plt.scatter(pos[:,0],pos[:,1])
    plt.axis([0,1,0,1])
    while (t<T): #Simulate from 0 to T
        pos,vel = take_step_RK4(dt,pos,vel,m)
        try:
            plt.clf()
            plt.scatter(pos[:,0],pos[:,1])
            plt.axis([0,1,0,1])
            plt.pause(1e-4)
            
        except:
            break
          
        t += dt
