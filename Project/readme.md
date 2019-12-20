This is my submission for the particle-mesh method n-body simulation. My method of integration is leapfrog.

My code relies on numpy functions: np.histogramdd for the density mesh assignment, np.fft.rfftn/irfftn for the FFTs, and np.gradient for the numerical differentiation. After benchmarking, I found that my computations are currently bottlenecked by the FFTs. I tried a couple of other FFT libraries but saw minimal improvement. 

Part 1) A single particle does in fact remain motionless. A (boring) animation of this is found in Animations/stationary.webm.

Part 2) Particles prepared in a circular orbit continue to orbit: Animations/orbit.webm

Part 3) My code is able to handle 10^5 particles, but my animations (matplotlib) aren't. I provide an animation of the non-periodic/periodic case for 1000 particles, about the maximum I can animate, in Animations/collapse_nonperiodic.webm and Animations/collapse_periodic.webm.

Some interesting structures form in both cases! 

I tracked the total energy of 10^5 particles in both cases too (dt= 0.001, grid size = 100). 

In the Energy folder, I attached two screenshots tracking the total energy every 100 steps for 1000 steps of the simulation. 

In the periodic case, energy is not extremely well conserved but fluctuates within a couple percent. I believe these fluctuations are just due to the limited precision of the timestep and grid size, which could be made smaller but would force the simulation to run for a very very long time. 

In the non-periodic case 

Part 4) 
