import numpy as np
from matplotlib import pyplot as plt
import time as ti
#TODO: 3D
#TODO: Boundary conditions
#TODO: Optimize
#TODO: Questions


try: 
    del fpot 
except:
    print("POT NOT DELETED")
    pass
try: 
    del a1
except: 
    pass

G=1 #gravitationnal constant
ic = 'unit orbit' #initial conditions


if ic == 'stationary':
    n=1
    pos = np.array([[0.5,0.5,0.5]])
    vel = np.array([[0,0,0]])
    m = np.array([1])
elif ic == 'orbit':
    n=2
    pos = np.array(([[0.5,0.5,0.5],[0.75,0.5,0.5]]))
    vel = np.array(([[0,0,0],[0,2.82842712475,0]]))   #63.24
    m = np.array([2,.000001])
elif ic == 'unit orbit':
    n=2
    pos = np.array(([[0.5,0.5,0.5],[0.75,0.5,0.5]]))
    vel = np.array(([[0,0,0],[0,2,0]]))   #63.24
    m = np.array([1,.000001])
elif ic == 'collapse':
    n=2
    pos = np.array(([[0.5,0.5,0.5],[0.75,0.5,0.5]]))
    vel = np.array(([[0,0,0],[0,0,0]]))
    m = np.array([1,1])
else:
    n=100000
    pos = np.random.rand(n,3)
    vel = (np.random.rand(n,3)*2-1)*0
    m = np.random.rand(n)
    
def gravity(p1,p2,m1,m2,epsilon=0.05,pot=False):
    x1=p1[0];x2=p2[0];y1=p1[1];y2=p2[1];z1=p1[2];z2=p2[2]
    num = G*m1*m2
    if not pot:
        den = ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2+epsilon**2)**(3/2)
        return np.array([-(num/den)*(x1-x2),-(num/den)*(y1-y2),-(num/den)*(z1-z2)])
    else:
        den = np.sqrt(((x1-x2)**2+(y1-y2)**2+(z1-z2)**2+epsilon**2))
        return num/den
    
def cal_force(r,m,epsilon=0.05,pot=False): #calculate force between 2 particles
    nparticles = len(r)
    forces=np.zeros((nparticles,3))
    V=0
    for i in range(nparticles):
        for j in range(nparticles):
            if (i!=j):
                if pot:
                    V-=gravity(r[i],r[j],m[i],m*np.array([m,m]).transpose()[j],epsilon=epsilon,pot=True)
                else:
                    forces[i]=gravity(r[i],r[j],m[i],m[j],epsilon=epsilon) #here just the gravity
    if pot:
        return V
    else:
        return forces

def cal_energy(r,v,m,epsilon=0.05):
     T = np.sum(0.5*m*(v[:,0]**2+v[:,1]**2+v[:,2]**2))
     V = cal_force(r,m,pot=True)
     return T+V

def cal_force_mesh(r,m,grid=128,eps=0.05,periodic=False):  
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
        density,(by,bx,bz) = np.histogramdd(pos,bins=grid,range=[[0, 1], [0, 1],[0, 1]],weights=m)
    density = density*(grid**3)
    #t2=ti.time()
    if not 'fpot' in globals():
        print("Computing Kernel")
        global fpot
        xs = np.linspace(-bgrid//2,bgrid//2,num=bgrid+1)[:-1]
        r2 = np.zeros((bgrid,bgrid,bgrid))
        for i in range(bgrid):
            for j in range(bgrid):
                for k in range(bgrid):
                    r2[i,j,k]=(xs[i]**2)+(xs[j]**2)+(xs[k]**2)
        r = np.sqrt(r2+(grid*eps)**2)
        del r2
        pot = -G/r
        del r
        pot = np.fft.fftshift(pot)
        fpot = np.fft.rfftn(pot)
        #fpot = np.fft.fftn(pot)
        print ("Kernel Finished")
    #t3= ti.time()
    phi = np.fft.irfftn(np.fft.rfftn(density)*fpot)
    #phi = np.real(np.fft.ifftn(np.fft.fftn(density)*fpot,s=(bgrid,bgrid,bgrid)))
    #phi = np.fft.fftshift(phi)
    #t4 = ti.time()
    #t5 = ti.time()
    xgrad,ygrad,zgrad = np.gradient(-phi)
    forces = np.empty((n,3))
    digx = np.digitize(x,bx)-1
    digy = np.digitize(y,by)-1
    digz = np.digitize(z,bz)-1
    #digx[digx==grid] = grid-1; digy[digy==grid] = grid-1; digz[digz==grid] = grid-1
    forces[:,0] = xgrad[digx,digy,digz]
    forces[:,1] = ygrad[digx,digy,digz]
    forces[:,2] = zgrad[digx,digy,digz]
    """
    #plt.figure(); plt.imshow(np.sqrt(xgrad**2)); plt.title("Forcex Magnitude");
    #plt.figure(); plt.imshow(np.sqrt(ygrad**2)); plt.title("Forcey Magnitude");
    plt.figure(); plt.imshow(density[:,:,64]); plt.title("Density"); 
    plt.figure(); plt.imshow(phi[:,:,64]); plt.title("Potential Field"); 
    plt.figure(); plt.imshow(pot[:,:,64]); plt.title("Per-Particle Potential"); 
    """
    #t6 = ti.time()
    #print("Gridding:" + str(t2-t1))
    #print("FFT:" + str(t4-t3))
    #print("Force:" + str(t6-t5))
    return forces*np.array([m,m,m]).transpose()/grid
    

def take_step_RK4(dt,pos,v,m,cal_fn=cal_force_mesh,periodic=False):   
    n = np.shape(m)[0]       
    #ntegration of d2y/dt2=f, here f is the acceleration
    x=pos[:,0];y=pos[:,1];z=pos[:,2];vx=v[:,0];vy=v[:,1];vz=v[:,2]
    em  = np.array([m,m,m]).transpose()
    pos1=np.empty((n,3))
    pos1[:,0]=x
    pos1[:,1]=y
    pos1[:,2]=z
    f1=cal_fn(pos1, m)/em
    x2=x+(dt/2)*vx
    y2=y+(dt/2)*vy
    z2=z+(dt/2)*vz
    pos2=np.empty((n,3))
    pos2[:,0]=x2
    pos2[:,1]=y2
    pos2[:,2]=z2
    f2=cal_fn(pos2,m)/em
    x3=x+(dt/2)*vx+(dt*dt/4)*f1[:,0]
    y3=y+(dt/2)*vy+(dt*dt/4)*f1[:,1]
    z3=z+(dt/2)*vz+(dt*dt/4)*f1[:,2]
    pos3=np.empty((n,3))  
    pos3[:,0]=x3
    pos3[:,1]=y3
    pos3[:,2]=z3
    f3=cal_fn(pos3,m)/em
    x4=x+(dt)*vx+(dt*dt/2)*f2[:,0]
    y4=y+(dt)*vy+(dt*dt/2)*f2[:,1]
    z4=z+(dt)*vz+(dt*dt/2)*f2[:,2]
    pos4=np.empty((n,3))
    pos4[:,0]=x4
    pos4[:,1]=y4
    pos4[:,2]=z4
    f4=cal_fn(pos4,m)/em
    f = f1+2*f2+2*f3+f4
    vxfinal=vx+(dt/6)*f[:,0]
    vyfinal=vy+(dt/6)*f[:,1]
    vzfinal=vz+(dt/6)*f[:,2]
    f = f1+f2+f3
    xfinal=x+dt*vx+(dt*dt/6)*f[:,0]
    yfinal=y+dt*vy+(dt*dt/6)*f[:,1]
    zfinal=z+dt*vz+(dt*dt/6)*f[:,2]
    posfinal=np.empty((len(x),3))
    vfinal=np.empty((len(x),3))
    posfinal[:,0]=xfinal
    posfinal[:,1]=yfinal
    posfinal[:,2]=zfinal
    vfinal[:,0]=vxfinal
    vfinal[:,1]=vyfinal
    vfinal[:,2]=vzfinal
    return posfinal,vfinal

def take_step_leapfrog(dt,pos,v,m,cal_fn=cal_force_mesh,periodic=False):  
    em = np.array([m,m,m]).transpose()
    #ntegration of d2y/dt2=f, here f is the acceleration
    if not 'a1' in globals():
        global a1
        a1=cal_fn(pos, m)/em
    pos = pos + v*dt + 0.5*a1*dt**2
    a2 = cal_fn(pos,m)/em 
    v = v + 0.5*(a1+a2)*dt
    a1 = a2
    return pos, v
    
    
if __name__ == '__main__':
    cal_fn = cal_force_mesh
    step_fn = take_step_leapfrog
    T=1
    t=0
    dt=0.01
    plot = 1
    i=0
    N = int(T/dt)
    xs = np.zeros((N+2,n))
    ys = np.zeros((N+2,n))
    zs = np.zeros((N+2,n))
    """
    if plot:
        plt.clf()
        plt.scatter(pos[:,0],pos[:,1])
        plt.axis([0,1,0,1])
    """
    while (t<T): #Simulate from 0 to T
        #print(t)
        #try:
        pos,vel = step_fn(dt,pos,vel,m,cal_fn=cal_fn)
       # except:
        #    print("exception")
        #    break
        
        x = pos[:,0]; y=pos[:,1]; z=pos[:,2]
        """tationary
        xx = np.logical_or(x<0,x>1)   
        yy = np.logical_or(y<0,y>1)   
        zz = np.logical_or(z<0,z>1)
        a = np.logical_or(xx,yy)
        a = np.logical_or(zz,a)
        e = np.sum(a)
        n-=e
        emm = m.copy()
        if e>0:
            a = np.logical_not(a)
            pos = np.zeros((n,3)) 
            velc = np.zeros((n,3))
            m = np.zeros(n)
            pos[:,0]= x[a]
            pos[:,1]= y[a]
            pos[:,2]= z[a]
            velc[:,0] = vel[:,0][a]
            velc[:,1] = vel[:,1][a]
            velc[:,1] = vel[:,2][a]
            vel = velc
            m = emm[a]
        if n == 0:
            print("all particles have escaped")
            break
        #xs[i]=x; ys[i]=y; zs[i]=z
        """
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
        print("Step " + str(i) + "/" +str(N))
        if i%100==0:
            "Calculating Total Energy" 
            print("Total Energy: "+str(cal_energy(pos,vel,m)))

plt.figure(figsize=(5,5))

#xs = xs[xs!=0]; ys = ys[ys!=0]
plt.axis([0,1,0,1])
plt.axis('equal')
plt.plot(xs,ys)


"""
for i in range(len(xs.transpose())):
    plt.scatter(xs.transpose()[i],ys.transpose()[i])
    plt.plot(xs.transpose()[i],ys.transpose()[i])
    
"""