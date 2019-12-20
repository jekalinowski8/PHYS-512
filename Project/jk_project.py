import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import time as ti

try: 
    del fpot 
except:
    pass
try: 
    del a1
except: 
    pass

G=1 #gravitationnal constant
ic = 'q4' #initial conditions

#Parameters 
dt=0.0005
T=1
plot = 0


if not plot: 
    plt.ioff()
else: 
    plt.ion()

#Initial conditions
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
    

elif ic == 'q4': 
    periodic = True
    size = 10
    n = size**3
    print("Setting up initial conditions, q4")
    ks = np.zeros((size,size,size))
    for i in range(size):
        for j in range(size):
            for l in range(size):
                ks[i,j,l] = np.sqrt(i**2+j**2+l**2)
    ks[0,0,0] = 1
    k = ks**(-3/2)
    den = np.fft.irfftn(k)
    den = np.fft.fftshift(den)
    pos = np.zeros((n,3))
    vel = np.zeros((n,3))
    m = np.zeros(n)
    xs = np.linspace(0,1,num=size)
    print("Finished Setup")
    c = 0
    for i in range(size):
        for j in range(size):
            for l in range(size):
                pos[c] = np.array([xs[i],xs[j],xs[l]])
                m[c] = den[i,j,l]
                c+=1
    m = m*100
                
else:
    n=100000
    periodic = False
    pos = np.random.rand(n,3)
    vel = (np.random.rand(n,3)*2-1)*0
    m = np.random.rand(n)
    

#Calculate the total energy of a system
def cal_energy(r,v,m,periodic=False,epsilon=0.05):
     T = np.sum(0.5*m*(v[:,0]**2+v[:,1]**2+v[:,2]**2))
     V = cal_force_mesh(r,m,periodic=periodic,calE=True)
     return T+V
    
#Calculate the forces between particles using a particle mesh
def cal_force_mesh(r,m,grid=100,eps=0.05,periodic=False,calE=False):  
    #t1 = ti.time()64
    x = r[:,0]; y = r[:,1]; z=r[:,2]
    n = len(x)
    #Density grid assignment in periodic/non-periodic
    if not periodic: 
        bgrid = 2*grid
        density,(bx,by,bz) = np.histogramdd(pos,bins=bgrid,range=[[0, 2], [0, 2],[0, 2]],weights=m)
        density[grid:,:,:]=0
        density[:,grid:,:]=0
        density[:,:,grid:]=0
    else: 
        bgrid = grid
        density,(by,bx,bz) = np.histogramdd(pos,bins=bgrid,range=[[0, 1], [0, 1],[0, 1]],weights=m)
    
    #Compute the kernel only once
    if not 'fpot' in globals(): 
        print("Computing Kernel")
        global fpot
        xs = np.linspace(-grid,grid,bgrid+1)[:-1]
        r2 = np.zeros((bgrid,bgrid,bgrid))
        for i in range(bgrid):
            for j in range(bgrid):
                for k in range(bgrid):
                    r2[i,j,k]=(xs[i]**2)+(xs[j]**2)+(xs[k]**2)
        r = np.sqrt(r2+(grid*eps)**2)
        pot = -G/r
        pot = np.fft.fftshift(pot)
        #fpot =  pyfftw.interfaces.numpy_fft.rfftn(pot)
        fpot = np.fft.rfftn(pot)
        print ("Kernel Finished")
    density = density*(grid**3)
    #t2=ti.time()
    #t3 = ti.time()
    #Convololve density with the kernel
    phi =  np.fft.irfftn(np.fft.rfftn(density)*fpot)
    #t4 = ti.time()
    #Compute the gradient of potential, force, with second order central difference
    xgrad,ygrad,zgrad = np.gradient(-phi)
    forces = np.empty((n,3))
    #Deal with the edge cases 
    digx = np.digitize(x,bx)-1
    digy = np.digitize(y,by)-1
    digz = np.digitize(z,bz)-1
    digx[digx>=grid]=grid-1        
    digy[digy>=grid]=grid-1
    digz[digz>=grid]=grid-1
    a = np.logical_or(np.logical_or(x>1,y>1),z>1)
    b = np.logical_or(np.logical_or(x<0,y<0),z<0)
    a = np.logical_or(a,b)
    forces[:,0] = xgrad[digx,digy,digz]
    forces[:,1] = ygrad[digx,digy,digz]
    forces[:,2] = zgrad[digx,digy,digz]
    forces[:,0][a] = 0; forces[:,1][a] = 0; forces[:,2][a] = 0;
    #t5 = ti.time()
    #print("Grid Time:" + str(t2-t1))
    #print("FFT Time:" + str(t4-t3))
    #print("Force Time:" + str(t5-t4))
    if calE: 
        return np.sum(m*phi[digx,digy,digz])
    else:
        return forces*np.array([m,m,m]).transpose()/grid

#Leapfrog integration    
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
    
#Run through the simulation, animate
cal_fn = cal_force_mesh
step_fn = take_step_leapfrog
scatters = []
data = []
t=0
i=0
N = int(T/dt)
xs = np.zeros((N+2,n))
ys = np.zeros((N+2,n))
zs = np.zeros((N+2,n))

if ic == 'orbit':
    xvals = np.linspace(0.25,0.75,num=1000)
    yvals1 = 0.25*(2-np.sqrt(-16*xvals**2+16*xvals-3))
    yvals2 = 0.25*(2+np.sqrt(-16*xvals**2+16*xvals-3))
if plot:
    plt.figure(figsize=(5,5))
while (t<T): #Simulate from 0 to T
    t1 = ti.time()
    pos,vel = step_fn(dt,pos,vel,m,cal_fn=cal_fn,periodic=periodic)
    if periodic: 
        pos = np.mod(pos,1)
    x = pos[:,0]; y=pos[:,1]; z=pos[:,2]
    xs[i]=x; ys[i]=y; zs[i]=z     
    data.append(np.array([x,y,z]).transpose())    
    i+=1
    t += dt
    t2 = ti.time()
    
    if i%100==0:
        print("Total Energy: "+str(cal_energy(pos,vel,m,periodic=periodic)))
        #print("Step " + str(i) + "/" +str(N) +";" + " Calculation Time per Iteration: "+ str(t2-t1))


print("Animating:")
fig = plt.figure(figsize=(10,10))
ax = p3.Axes3D(fig)
scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]
iterations = N
def animate_scatters(iteration,data, scatters):
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
    return scatters
#ax.view_init(25, 10)
ax.set_xlim3d([0,1])
ax.set_xlabel('X')
ax.set_ylim3d([0,1])
ax.set_ylabel('Y')
ax.set_zlim3d([0,1])
ax.set_zlabel('Z')
ax.set_title('1000 Particles, Periodic')
if ic == 'orbit':
    ax.plot(xvals,yvals1,0.5,color='k',ls='--',zorder=20)
    ax.plot(xvals,yvals2,0.5,color='k',ls='--',zorder=21)      
ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters), interval=50, blit=False, repeat=True)
print("Done Animating")
#mywriter = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
#ani.save('orbit1.mp4', writer=mywriter)
#print("Animation created")
plt.show()


