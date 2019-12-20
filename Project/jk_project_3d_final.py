import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import time as ti
plt.rcParams['animation.ffmpeg_path'] = './ffmpeg'

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

dt=0.001
T=3
plot = 0

if not plot: 
    plt.ioff()
else: 
    plt.ion()

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
    


def cal_energy(r,v,m,periodic=False,epsilon=0.05):
     T = np.sum(0.5*m*(v[:,0]**2+v[:,1]**2+v[:,2]**2))
     V = cal_force_mesh(r,m,periodic=periodic,calE=True)
     return T+V
     
def cal_force_mesh(r,m,grid=100,eps=0.05,periodic=False,calE=False):  
    #t1 = ti.time()
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
    #phi =  pyfftw.interfaces.numpy_fft.irfftn( pyfftw.interfaces.numpy_fft.rfftn(density)*fpot)
    phi =  np.fft.irfftn(np.fft.rfftn(density)*fpot)
    #t4 = ti.time()
    xgrad,ygrad,zgrad = np.gradient(-phi)
    forces = np.empty((n,3))
    digx = np.digitize(x,bx)-1
    digy = np.digitize(y,by)-1
    digz = np.digitize(z,bz)-1
    digx[digx>=grid]=grid-1        
    digy[digy>=grid]=grid-1
    digz[digz>=grid]=grid-1
    a = np.logical_or(np.logical_or(digx>=grid,digy>=grid),digz>=grid)
    b = np.logical_or(np.logical_or(digx==0,digy==0),digz==0)
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
scatters = []
data = []
t=0
i=0
N = int(T/dt)
xs = np.zeros((N+2,n))
ys = np.zeros((N+2,n))
zs = np.zeros((N+2,n))
fig = plt.figure()
ax = p3.Axes3D(fig)
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
    print("Progress: " + str(i) + "/" +str(N))
    if i%100==0:
        print("Total Energy: "+str(cal_energy(pos,vel,m,periodic=periodic)))
        print("Step " + str(i) + "/" +str(N) +";" + " Calculation Time per Iteration: "+ str(t2-t1))
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs,ys,zs)
if ic == 'orbit':
    ax.plot(xvals,yvals1,0.5)
    ax.plot(xvals,yvals2,0.5)
ax.set_zlabel('x')
ax.set_zlabel('y')
ax.set_zlabel('z')
ax.set_xlim3d(0.05,1)
ax.set_ylim3d(0.05,1)
ax.set_zlim3d(0.05,1)
plt.show()
"""
print("Animating:")
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
ax.set_title('Orbit')
if ic == 'orbit':
    ax.plot(xvals,yvals1,0.5,color='k',ls='--',zorder=20)
    ax.plot(xvals,yvals2,0.5,color='k',ls='--',zorder=21)      
ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters), interval=50, blit=False, repeat=True)
print("Done Animating")
mywriter = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
#ani.save('orbit1.mp4', writer=mywriter)
#print("Animation created")
plt.show()



