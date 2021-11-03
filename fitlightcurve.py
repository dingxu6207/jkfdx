# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:11:41 2021

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
import torch
import pickle,time
import numpy as np
from MLPnet_reg import NET as NET 
from scipy import interpolate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
# num_epochs = 10000000 #最大迭代次数
learning_rate = 1e-5 #初始学习率
inputCH=4#输入参数个数
outputCH=100 #回归参数个数
convNUM=500 #隐藏层卷积核个数，可修改
layers=2#残差块数量，每块有两个卷积层，可修改
netfile='dx_m_new.mod' #模型文件名

model_state_dict = torch.load(netfile,map_location='cpu').state_dict()
model = NET(inputCH,convNUM,layers,outputCH).to(device)
model.load_state_dict(model_state_dict)
model.eval() #为了防止BN层和dropout的随机性，直接用evaluation方式训练


path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 10389809.txt'
data = np.loadtxt(path+file)
phrase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])
sx1 = np.linspace(0,1,100)
sy1 = np.interp(sx1,phrase,datay)

x = np.copy(sx1)
noisy = np.copy(sy1)


plt.figure(10)
plt.plot(x,noisy,'.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

nwalkers = 30
niter = 100
init_dist = [(55.,75.),(1,6),(0,0.2),(0.9,1.0)]
ndim = len(init_dist)
sigma = np.ones(len(noisy))*np.diff(noisy,2).std()/np.sqrt(6)

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

    incl = z[0]
    q = z[1]
    f = z[2]
    t2t1 = z[3]
    allpara = [incl,q,f,t2t1]
    npallpara = np.array(allpara) 
    npallpara = npallpara.astype('float32')
    Input=torch.from_numpy(npallpara).to(device) 
    output = model(Input) #训练！
    output = output.cpu().detach().numpy() #预测值
    #output[0] = output[0]-np.mean(output[0])
    lnp = -0.5*np.sum((output-noisy)**2/sigma**2)
    
    return lnp

tempsigma = []
datatemp = []
def run(init_dist, nwalkers, niter):
    
    ndim = len(init_dist)
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

pos = run(init_dist, nwalkers, niter)
nptemp = np.array(datatemp).T
#figure = corner.corner(nptemp,bins=50,quantiles=[0.1, 0.25, 0.5, 0.75],labels=[r"$incl$", r"$q$", r"$f_0$", r"$t2t1$"],
#                       label_kwargs={"fontsize": 15},show_titles=True, title_kwargs={"fontsize": 15}, color ='blue')
figure = corner.corner(nptemp,bins=50,labels=[r"$incl$", r"$q$", r"$f_0$", r"$t2t1$"],
                       label_kwargs={"fontsize": 15},show_titles=True, title_kwargs={"fontsize": 15}, color ='blue')
figure.savefig("corner.png")