# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:16:40 2021

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

path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 10389809.txt'
data = np.loadtxt(path+file)
phrase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])
flux = datay
sx1 = np.linspace(0,1,100)
s = np.diff(datay,2).std()/np.sqrt(6)
num = len(datay)
func1 = interpolate.UnivariateSpline(data[:,0], datay,s=s*s*num)#强制通过所有点
sy1 = func1(sx1)


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
z = [65.75,3.38,0.10,0.96]
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


plt.figure(10)
plt.plot(output,'.')
plt.plot(sy1,'.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向