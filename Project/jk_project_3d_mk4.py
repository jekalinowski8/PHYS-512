import numpy as np
from numpy import fft 
from matplotlib import pyplot as plt
import time as ti
from numba import jit
from matplotlib import animation as animation
from mpl_toolkits.mplot3d import Axes3D
#TODO: Questions


try: 
    del fpot 
except:
    pass
try: 
    del a1
except: 
    pass

G=1 #gravitationnal constant
ic = 'orbit' #initial conditions

dt=0.01
T=3
plot = 1




if ic == 'stationary':
    n=1
    pos = np.array([[0.5,0.5,0.5]])
    vel = np.array([[0,0,0]])
    m = np.array([1])
    periodic = False
    
elif ic == 'orbit':
    n=2
    pos = np.array(([[0.5,0.5,0.5],[0.75,0.5,0.5]]))
    vel = np.array(([[0,0,0],[0,2.82842712475,0]]))   #63.24
    m = np.array([2,.000001])
    periodic = False
    
elif ic == 'ptest': 
    n = 2
    pos = np.array([[0.9,0.9,0.9],[0.1,0.1,0.1]])
    vel = np.array([[0,0,0],[0,0,0]])
    m = np.array([1,1])
    periodic = True
    
else:
    n=100000
    pos = np.random.rand(n,3)
    vel = (np.random.rand(n,3)*2-1)*0
    m = np.random.rand(n)
    
    
    
@jit(nopython=True,fastmath=True,parallel=True)
def compute_kernel(bgrid=256,eps=0.05):
    print("Computing Kernel")
    bgrid = np.int64(bgrid)
    grid = np.int64((bgrid//2))
    xs = np.linspace(-grid,grid,np.int64(bgrid+1))[:-1]
    r2 = np.zeros((bgrid,bgrid,bgrid))
    for i in range(bgrid):
        for j in range(bgrid):
            for k in range(bgrid):
                r2[i,j,k]=(xs[i]**2)+(xs[j]**2)+(xs[k]**2)
    r = np.sqrt(r2+(grid*eps)**2)
    pot = -G/r
    pot = fft.fftshift(pot)
    fpot = fft.rfftn(pot)
    print ("Kernel Finished")
    return fpot
fpot = compute_kernel()

def cal_energy(r,v,m,periodic=False,epsilon=0.05):
     T = np.sum(0.5*m*(v[:,0]**2+v[:,1]**2+v[:,2]**2))
     V = cal_force_mesh(r,m,periodic=periodic,calE=True)
     return T+V
     
@jit(nopython=True,fastmath=True,parallel=True)
def cal_force_mesh(r,m,grid=128,eps=0.05,periodic=False,calE=False):  
    t1 = ti.time()
    x = r[:,0]; y = r[:,1]; z=r[:,2]
    n = len(x)
    if not periodic: 
        bgrid = 2*grid
        density,(bx,by,bz) = np.histogramdd(pos,bins=bgrid,range=[[0, 2], [0, 2],[0, 2]],weights=m)
        density[grid:,:,:]=0
        density[:,grid:,:]=0
        density[:,:,grid:]=0
    else: 
        bgrid = grid
        density,(by,bx,bz) = np.histogramdd(pos,bins=bgrid,range=[[0, 1], [0, 1],[0, 1]],weights=m)
    density = density*(grid**3)
    t2=ti.time()
    t3 = ti.time()
    phi = np.fft.irfftn(np.fft.rfftn(density)*fpot)
    t4 = ti.time()
    xgrad,ygrad,zgrad = np.gradient(-phi)
    forces = np.empty((n,3))
    digx = np.digitize(x,bx)-1
    digy = np.digitize(y,by)-1
    digz = np.digitize(z,bz)-1
    digx[digx>=grid]=grid-1
    digy[digy>=grid]=grid-1
    digz[digz>=grid]=grid-1
    a = np.logical_or(np.logical_or(digx>=grid,digy>=grid),digz>=grid)
    forces[:,0] = xgrad[digx,digy,digz]
    forces[:,1] = ygrad[digx,digy,digz]
    forces[:,2] = zgrad[digx,digy,digz]
    forces[:,0][a] = 0; forces[:,1][a] = 0; forces[:,2][a] = 0;
    t5 = ti.time()
    print("Grid Time:" + str(t2-t1))
    print("FFT Time:" + str(t4-t3))
    print("Force Time:" + str(t5-t4))
    if calE: 
        return np.sum(m*phi[digx,digy,digz])
    else:
        return forces*np.array([m,m,m]).transpose()/grid
    
def take_step_leapfrog(dt,pos,v,m,cal_fn=cal_force_mesh,periodic=False):  
    em = np.array([m,m,m]).transpose()
    #ntegration of d2y/dt2=f, here f is the acceleration
    if not 'a1' in globals():
        global a1
        a1=cal_fn(pos, m,periodic=periodic)/em
    pos = pos + v*dt + 0.5*a1*dt**2
    a2 = cal_fn(pos,m,periodic=periodic)/em 
    v = v + 0.5*(a1+a2)*dt
    a1 = a2
    return pos, v
    

cal_fn = cal_force_mesh
step_fn = take_step_leapfrog


t=0
i=0
N = int(T/dt)
xs = np.zeros((N+2,n))
ys = np.zeros((N+2,n))
zs = np.zeros((N+2,n))
while (t<T): #Simulate from 0 to T
    t1 = ti.time()
    pos,vel = step_fn(dt,pos,vel,m,cal_fn=cal_fn,periodic=periodic)
    x = pos[:,0]; y=pos[:,1]; z=pos[:,2]
    xs[i]=x; ys[i]=y; zs[i]=z
    
    if plot:
        try:
            plt.clf()
            plt.scatter(x,y)
            plt.axis([0,1,0,1])
            plt.pause(1e-2)
        except:
            break
    i+=1
    t += dt
    
    if i%100==0:
        print("Total Energy: "+str(cal_energy(pos,vel,m,periodic=periodic)))
    t2 = ti.time()
   # print("Step " + str(i) + "/" +str(N) +";" + " Calculation Time per Iteration: "+ str(t2-t1))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs,ys,zs)
ax.set_zlabel('x')
ax.set_zlabel('y')
ax.set_zlabel('z')
ax.set_xlim3d(0.01,1)
ax.set_ylim3d(0.01,1)
ax.set_zlim3d(0.01,1)
plt.show()



