
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 08:43:35 2018

@author: jkf
"""
from matplotlib import pyplot as plt
import torch

import pickle,time
import numpy as np

from MLPnet_reg import NET as NET 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# num_epochs = 10000000 #最大迭代次数
learning_rate = 1e-5 #初始学习率
inputCH=4#输入参数个数
outputCH=100 #回归参数个数
convNUM=500 #隐藏层卷积核个数，可修改
layers=2#残差块数量，每块有两个卷积层，可修改

# curr_lr=learning_rate  #当前的学习率
# K=0 #循环显示的初始参数

netfile='dx_m_new.mod' #模型文件名

#dat0=np.loadtxt('savedata01020.txt')
dat0=pickle.load(open('savedata01020.dat','rb'))
D=((dat0[:,100]>50)* (dat0[:,0]<10))>0
dat0=dat0[D]

dat0[:,:100]=-2.5*np.log10(dat0[:,:100])
dat0mean=dat0[:,:100].mean(axis=1)
dat0[:,:100]=dat0[:,:100]-dat0mean[:,np.newaxis]
print('finished reading data')


ID=1234 #选择一个样本
para=dat0[[ID],100:104].astype('float32') #输入参数
Input=torch.from_numpy(para).to(device) 


Label=dat0[[ID],:100].astype('float32') #标签


#调入模型   
model_state_dict = torch.load(netfile,map_location='cpu').state_dict()
model = NET(inputCH,convNUM,layers,outputCH).to(device)
model.load_state_dict(model_state_dict)
    
model.eval() #为了防止BN层和dropout的随机性，直接用evaluation方式训练

#预测
t1=time.time()
output = model(Input) #训练！
output = output.cpu().detach().numpy() #预测值
print('running time:', time.time()-t1,'seconds')  


#画图
ax5=plt.subplot(211)
ax6=plt.subplot(212)
ax5.plot(Label[0],'-')
ax5.plot(output[0],'.')
ax6.plot(Label[0]-output[0],'-')
ax5.set_title(str(para[0]))

ax6.set_title((Label[0]-output[0]).std())

 