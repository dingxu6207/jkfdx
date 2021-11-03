# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:27:26 2021

@author: dingxu
"""

import numpy as np
import corner
import emcee
import matplotlib.pyplot as pl
from matplotlib.pyplot import cm 
import matplotlib.mlab as mlab
from scipy.stats import norm
import matplotlib.pyplot as plt

#def q(a,b,c,x):
#    quad = a*x**2. + b*x + c
#    return quad
#
#x = np.linspace(-1,1,101)
#
#quad = q(1.,0.,-0.2,x)
#noise = np.random.normal(0.0, 0.1, quad.shape)
#noisy = quad + noise
#plt.figure(0)
#pl.plot (x,noisy,"k.")
#pl.show()

def q(fbg,f0, rc, x):
    return fbg+f0/(1+(x/rc)**2)

data = np.loadtxt('arraylist.txt')
x = data[0,:]
noisy = data[1,:]


nwalkers = 30
niter = 1000
init_dist = [(4.,15.),(48,88),(2,4)]
ndim = len(init_dist)
sigma = np.diff(noisy,2).std()/np.sqrt(6)
#sigma = 0.05

priors = init_dist

def rpars(init_dist):
    return [np.random.rand() * (i[1]-i[0]) + i[0] for i in init_dist]


def lnprior(priors, values):
    
    lp = 0.
    for value, prior in zip(values, priors):
        if value >= prior[0] and value <= prior[1]:
            lp+=0
        else:
            lp+=-np.inf 
    return lp


def lnprob(z):
    
    lnp = lnprior(priors,z)
    if not np.isfinite(lnp):
            return -np.inf

    # make a model using the values the sampler generated
    model = q(z[0],z[1],z[2],x)

    # use chi^2 to compare the model to the data:
    chi2 = 0.
    for i in range (len(x)):
            chi2+=((noisy[i]-model[i])**2)/(sigma**2)

    # calculate lnp
    lnprob = -0.5*chi2 + lnp

    return lnprob

tempsigma = []
datatemp = []
def run(init_dist, nwalkers, niter, ndim):

    # Generate initial guesses for all parameters for all chains
    p0 = np.array([rpars(init_dist) for i in range(nwalkers)])
    #print(p0)

    # Generate the emcee sampler. Here the inputs provided include the 
    # lnprob function. With this setup, the first parameter
    # in the lnprob function is the output from the sampler (the paramter 
    # positions).
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob)

    pos, prob, state = sampler.run_mcmc(p0, niter)

    for i in range(ndim):
        pl.figure(i+1)
        y = sampler.flatchain[:,i]
        n, bins, patches = pl.hist(y, 200, density=1, color="b", alpha=0.45)
        pl.title("Dimension {0:d}".format(i))
        
        mu = np.average(y)
        tempsigma.append(mu)
        sigma = np.std(y)  
        tempsigma.append(sigma)
        print ("mu,", "sigma = ", mu, sigma)
        
        datatemp.append(y)
        bf = norm.pdf(bins, mu, sigma)
        l = pl.plot(bins, bf, 'k--', linewidth=2.0)
        
    pl.show()
    return pos,tempsigma

'''
niter=10
pos = run(init_dist, nwalkers, niter, ndim)


color=cm.rainbow(np.linspace(0,1,nwalkers))
for i,c in zip(range(nwalkers),color):
    
    model = pos[-1-i,0]*x**2 + pos[-1-i,1]*x + pos[-1-i,2]
    
   
    plt.figure(3)
    pl.plot(x,model,c=c)    
pl.plot(x,noisy,"k.")
pl.xlabel("x")
pl.ylabel("f(x)")
pl.show()


'''
niter = 500

pos,tempsigma = run(init_dist, nwalkers, niter, ndim)

nptemp = np.array(datatemp).T
plt.figure(5)
figure = corner.corner(nptemp,bins=50,quantiles=[0.16, 0.5, 0.84],labels=[r"$f_b$", r"$f_0$", r"$R_c$"],
                       label_kwargs={"fontsize": 15},show_titles=True, title_kwargs={"fontsize": 15}, color ='blue')

figure.savefig("corner.png")
'''
color=cm.rainbow(np.linspace(0,1,nwalkers))
for i,c in zip(range(nwalkers),color):
    
    model = pos[-1-i,0]+pos[-1-i,1]/(1+(x/pos[-1-i,2])**2)
    
    plt.figure(4)
    pl.plot(x,model,c=c)
'''
yerr = 1/np.sqrt(noisy)
plt.figure(4)
model = tempsigma[0]+tempsigma[2]/(1+(x/tempsigma[4])**2)   
pl.plot(x,model)
pl.plot(x,noisy,"k.")
pl.xlabel("R(arcmin)")
pl.ylabel("Density(stars arcmin-2)")
plt.title('Auner_1')
plt.axvline(x=0.59,ls="--",c="green",linewidth=1.2)#添加垂直直线
plt.axvline(x=6.31,ls="--",c="green",linewidth=1.2)#添加垂直直线
plt.text(1.02, 57.83, 'Rc = 0.59', fontsize=12, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签
plt.text(6.4, 15.7, 'Rlim = 6.31', fontsize=12, color = "b", style = "italic", weight = "light", verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签
plt.savefig('Auner_1.png')

f0 = tempsigma[2]
sigb = tempsigma[1]
rc = tempsigma[4]
#sigb = 0.26
print('Rlim = ', rc*np.sqrt(f0/(3*sigb) - 1))