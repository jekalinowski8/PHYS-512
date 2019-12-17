import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
#TODO: 3D
#TODO: Boundary conditions
#TODO: Optimize
#TODO: Questions

G=1 #gravitationnal constant
ic = 'orbit' #initial conditions


if ic == 'stationary':
    n=1
    pos = np.array([[0.5,0.5]])
    vel = np.array([[0,0]])
    m = np.array([1])
elif ic == 'orbit':
    n=2
    pos = np.array(([[0.5,0.5],[0.75,0.5]]))
    vel = np.array(([[0,0],[0,2]]))   #63.24
    m = np.array([1,.0001])
elif ic == 'collapse':
    n=2
    pos = np.array(([[0.5,0.5],[0.75,0.5]]))
    vel = np.array(([[0,0],[0,1]]))
    m = np.array([1,1])
else:
    n=232
    pos = np.random.rand(n,2)
    vel = (np.random.rand(n,2)*2-1)*0
    m = np.random.rand(n)
    
def gravity(p1,p2,m1,m2,epsilon=0.05,pot=False):
    x1=p1[0];x2=p2[0];y1=p1[1];y2=p2[1]
    num = G*m1*m2
    if not pot:
        den = ((x1-x2)**2+(y1-y2)**2+epsilon**2)**(3/2)
        return np.array([-(num/den)*(x1-x2),-(num/den)*(y1-y2)])
    else:
        den = np.sqrt(((x1-x2)**2+(y1-y2)**2+epsilon**2))
        return num/den

def cal_force(x,m,epsilon=0.05,pot=False): #calculate force between 2 particles
   
    nparticles = len(x)
    forces=np.zeros((nparticles,2))
    V=0
    for i in range(nparticles):
        for j in range(nparticles):
            if (i!=j):
                if pot:
                    V-=gravity(x[i],x[j],m[i],m[j],epsilon=epsilon,pot=True)
                else:
                    forces[i]=gravity(x[i],x[j],m[i],m[j],epsilon=epsilon) #here just the gravity
    if pot:
        return V
    else:
        return forces

def cal_energy(x,v,m,epsilon=0.05):
     T = np.sum(0.5*m*v[:,0]**2+v[:,1]**2)
     V = cal_force(x,m,pot=True)
     return T+V
     

def cal_force_mesh(r,m,grid=100,eps=0.05,periodic=False):  
    x = r[:,0]; y = r[:,1]
    if not periodic: 
        bgrid = 2*grid
        density,by,bx =np.histogram2d(x,y,bins=bgrid,range=[[0, 2], [0, 2]],weights=m)
        density[grid:,:]=0
        density[:,grid:]=0
    else: 
        bgrid = grid
        density,by,bx= np.histogram2d(x,y,bins=grid,range=[[0, 1], [0, 1]],weights=m)
    density = density*(grid**2)
    xs = np.linspace(-bgrid//2,bgrid//2,num=bgrid+1)[:-1]
    r2 = np.tile(xs**2,(bgrid,1))
    r = np.sqrt(r2+r2.transpose()+(grid)*eps**2) 
    pot = -G/r
    phi = np.real(np.fft.ifft2(np.fft.fft2(density)*np.fft.fft2(pot),s=(bgrid,bgrid)))
    phi = np.fft.fftshift(phi)
   # plt.imshow(phi)
    xgrad,ygrad = np.gradient(-phi)
   # plt.figure(3); plt.imshow(phi)pass

   #assert(1==0
    forces = np.empty((len(x),2))
    digx = np.digitize(x,bx)-1
    digy = np.digitize(y,by)-1
    #a = np.logical_and(digx<grid,digy<grid)
    #digx = digx[a]; digy = digy[a]
    forces[:,0] = xgrad[digx,digy]
    forces[:,1] = ygrad[digx,digy]
    """
    plt.figure(); plt.imshow(np.sqrt(xgrad**2)); plt.title("Forcex Magnitude"); plt.show()
    plt.figure(); plt.imshow(np.sqrt(ygrad**2)); plt.title("Forcey Magnitude"); plt.show()
    plt.figure(); plt.imshow(density); plt.title("Density"); plt.show() 
    plt.figure(); plt.imshow(phi); plt.title("Potential Field"); plt.show()
    plt.figure(); plt.imshow(pot); plt.title("Per-Particle Potential"); plt.show()
    
    assert 0
   
    print(forces)
    assert 0
    """
    return forces
    

def take_step_RK4(dt,pos,v,m,cal_fn=cal_force_mesh,periodic=False):   
    n = np.shape(m)[0]       
    '''Integration of d2y/dt2=f, here f is the acceleration'''
    x=pos[:,0];y=pos[:,1];vx=v[:,0];vy=v[:,1]
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
    pos4=np.empty((n,2))
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
    #return pos,v print(pos)


if __name__ == '__main__':
    cal_fn = cal_force_mesh
    T=1
    t=0
    dt=0.001
    plot = 1
    i=0
    xs = np.zeros((int(T/dt),n))
    ys = np.zeros((int(T/dt),n))
    if plot:
        plt.clf()
        plt.scatter(pos[:,0],pos[:,1])
        plt.axis([0,1,0,1])
    while (t<T): #Simulate from 0 to T
        #print(t)
        #try:
        pos,vel = take_step_RK4(dt,pos,vel,m,cal_fn=cal_fn)
       # except:
        #    print("exception")
        #    break
        x = pos[:,0]; y=pos[:,1]
        xx = np.logical_or(x<0,x>1)   
        yy = np.logical_or(y<0,y>1)    
        a = np.logical_or(xx,yy)
        e = np.sum(a)
        n-=e
        emm = m.copy()
        if e>0:
            a = np.logical_not(a)
            pos = np.zeros((n,2)) 
            velc = np.zeros((n,2))
            m = np.zeros(n)
            pos[:,0]= x[a]
            pos[:,1]= y[a]
            velc[:,0] = vel[:,0][a]
            velc[:,1] = vel[:,1][a]
            vel = velc
            m = emm[a]
        if n == 0:
            print("all particles have escaped")
            break
        xs[i]=x; ys[i]=y
        if plot:
            try:
                plt.clf()
                plt.scatter(x,y)
                plt.axis([0,1,0,1])
                plt.pause(1e-4)
            except:
                break
        i+=1
        t += dt
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