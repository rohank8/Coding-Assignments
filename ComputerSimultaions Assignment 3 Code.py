#!/usr/bin/env python
# coding: utf-8

# In[33]:


name = "Rohan Kadam"
student_number = "20334092"
print(name,student_number)

import numpy as np
import matplotlib.pyplot as plt
def f(t,x):
    return ((1+t)*x)+ 1 -3*t + t**2
trange = np.linspace(0,5,25)
xrange = np.linspace(-3,3,25)
T,X = np.meshgrid(trange,xrange)
dx = (np.ones(f(T,X).shape))
fig, ax = plt.subplots()
dyy = f(T,X)/(np.sqrt((dx**2 + f(T,X)**2)))
dxx = dx/(np.sqrt((dx**2+f(T,X)**2)))
plt.figure(figsize = (10,6))
plt.quiver(T,X,dxx,dyy,color='black')
plt.xlim(0,5)
plt.ylim(-3,3)
plt.title('Directional field for $\dfrac{dx}{dt}$ = (1+t)x + 1 - 3t + $t^2$', fontsize = 14)
plt.xlabel('t', fontsize = 18)
plt.ylabel('x', fontsize = 18)
plt.show()


# In[34]:


def f(t,x):
    return ((1+t)*x)+ 1 -3*t + t**2
trange = np.linspace(0,5,25)
xrange = np.linspace(-3,3,25)
T,X = np.meshgrid(trange,xrange)
dx = (np.ones(f(T,X).shape))
fig, ax = plt.subplots()
dyy = f(T,X)/(np.sqrt((dx**2 + f(T,X)**2)))
dxx = dx/(np.sqrt((dx**2+f(T,X)**2)))

#lets define all our methods now
def SimpleEuler(t,x,step):
    return x+ step*f(t,x)
step = 0.04; start = 0.0; end = 5.0
x_zero =0.0655
arranging = np.arange(start,end,step)
n = int((end-start)/step)
SimpleEulerl = np.zeros(n)

SimpleEulerl[0] = x_zero

for i in range(1,n):
    SimpleEulerl[i] = SimpleEuler(arranging[i-1], SimpleEulerl[i-1], step)
    

plt.figure(figsize = (10,6))
plt.quiver(T,X,dxx,dyy,color='black')
plt.plot(arranging,SimpleEulerl,color = 'red',label='Simple Euler method')
plt.xlim(0,5)
plt.ylim(-3,3)
plt.legend()
plt.title('Ordinary Differential Equation $\dfrac{dx}{dt}$ graphed using Simple Euler method for step size h = 0.04')
plt.ylabel('x', fontsize = 18)
plt.xlabel('t', fontsize = 18)


# In[36]:


def f(t,x):
    return ((1+t)*x)+ 1 -3*t + t**2
trange = np.linspace(0,5,25)
xrange = np.linspace(-3,3,25)
T,X = np.meshgrid(trange,xrange)
dx = (np.ones(f(T,X).shape))
fig, ax = plt.subplots()
dyy = f(T,X)/(np.sqrt((dx**2 + f(T,X)**2)))
dxx = dx/(np.sqrt((dx**2+f(T,X)**2)))

#lets define all our methods now
def SimpleEuler(t,x,step):
    return x+ step*f(t,x)
def ImprovedEuler(t,x,step):
    x_new = x + 0.5*step*( f(t,x) + f(t+step, x + step*f(t, x)) )
    return x_new
def RungeKutta(t,x,step):
    k1 = f(t,x)
    k2 = f(t + 0.5*step, x + 0.5*step*k1)
    k3 = f(t + 0.5*step, x + 0.5*step*k2)
    k4 = f(t + step, x + step*k3)
    x_new = x + step/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4) 
    return x_new
step = 0.04; start = 0.0; end = 5.0
x_zero =0.0655
arranging = np.arange(start,end,step)
n = int((end-start)/step)
SimpleEulerl = np.zeros(n)
ImprovedEulerl = np.zeros(n)
RungeKuttal =np.zeros(n)
SimpleEulerl[0] = x_zero
ImprovedEulerl[0] = x_zero
RungeKuttal[0] = x_zero
for i in range(1,n):
    SimpleEulerl[i] = SimpleEuler(arranging[i-1], SimpleEulerl[i-1], step)
    ImprovedEulerl[i] = ImprovedEuler(arranging[i-1], ImprovedEulerl[i-1], step)
    RungeKuttal[i] = RungeKutta(arranging[i-1], RungeKuttal[i-1], step)

plt.figure(figsize = (10,6))
plt.quiver(T,X,dxx,dyy,color='black')
plt.plot(arranging,SimpleEulerl,color = 'red',label='Simple Euler method')
plt.plot(arranging,ImprovedEulerl,color='blue',label='Improved Euler method')
plt.plot(arranging,RungeKuttal,color='green',label='Runge-Kutta method')
plt.xlim(0,5)
plt.ylim(-3,3)
plt.legend()
plt.title('Ordinary Differential Equation $\dfrac{dx}{dt}$ graphed using various methods for step size h = 0.04')
plt.ylabel('x', fontsize = 18)
plt.xlabel('t', fontsize = 18)


# In[26]:


def f(t,x):
    return ((1+t)*x)+ 1 -3*t + t**2
trange = np.linspace(0,5,25)
xrange = np.linspace(-3,3,25)
T,X = np.meshgrid(trange,xrange)
dx = (np.ones(f(T,X).shape))
fig, ax = plt.subplots()
dyy = f(T,X)/(np.sqrt((dx**2 + f(T,X)**2)))
dxx = dx/(np.sqrt((dx**2+f(T,X)**2)))

#lets define all our methods now
def SimpleEuler(t,x,step):
    return x+ step*f(t,x)
def ImprovedEuler(t,x,step):
    x_new = x + 0.5*step*( f(t,x) + f(t+step, x + step*f(t, x)) )
    return x_new
def RungeKutta(t,x,step):
    k1 = f(t,x)
    k2 = f(t + 0.5*step, x + 0.5*step*k1)
    k3 = f(t + 0.5*step, x + 0.5*step*k2)
    k4 = f(t + step, x + step*k3)
    x_new = x + step/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4) 
    return x_new
step = 0.02; start = 0.0; end = 5.0
x_zero =0.0655
arranging = np.arange(start,end,step)
n = int((end-start)/step)
SimpleEulerl = np.zeros(n)
ImprovedEulerl = np.zeros(n)
RungeKuttal =np.zeros(n)
SimpleEulerl[0] = x_zero
ImprovedEulerl[0] = x_zero
RungeKuttal[0] = x_zero
for i in range(1,n):
    SimpleEulerl[i] = SimpleEuler(arranging[i-1], SimpleEulerl[i-1], step)
    ImprovedEulerl[i] = ImprovedEuler(arranging[i-1], ImprovedEulerl[i-1], step)
    RungeKuttal[i] = RungeKutta(arranging[i-1], RungeKuttal[i-1], step)

plt.figure(figsize = (10,6))
plt.quiver(T,X,dxx,dyy,color='black')
plt.plot(arranging,SimpleEulerl,color = 'red',label='Simple Euler method')
plt.plot(arranging,ImprovedEulerl,color='blue',label='Improved Euler method')
plt.plot(arranging,RungeKuttal,color='green',label='Runge-Kutta method')
plt.xlim(0,5)
plt.ylim(-3,3)
plt.legend()
plt.title('Ordinary Differential Equation $\dfrac{dx}{dt}$ graphed using various methods for step size h = 0.02')
plt.ylabel('x', fontsize = 18)
plt.xlabel('t', fontsize = 18)

