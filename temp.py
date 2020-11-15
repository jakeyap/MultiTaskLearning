#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:13:10 2020

@author: jakeyap
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


x = torch.normal(1,1,(1,500))
y = torch.normal(1,1,(1,500))

x = x.reshape(-1,1)
y = y.reshape(-1,1)

x = x * 2.5 - 2
y = y * 1.5 + x * 0.5

fig = plt.figure()
plt.scatter(x,y)

plt.ylim([-10,10])
plt.xlim([-10,10])

model1 = nn.Linear(1, 10)
model2 = nn.ReLU()
model3 = nn.Linear(10, 2)

lossfn = nn.MSELoss().cuda()

optimizer1 = torch.optim.Adam(params=model1.parameters(), lr=0.005)
optimizer3 = torch.optim.Adam(params=model3.parameters(), lr=0.005)
EPOCHS = 50
colors = ['grey', 'green', 'blue', 'black', 'red']

def eval_mode():
    for eachmodel in [model1, model2, model3]:
        eachmodel.train()
        eachmodel.cuda()

def test_mode():
    for eachmodel in [model1, model2, model3]:
        eachmodel.eval()
        eachmodel.cuda()

x = x.cuda()
y = y.cuda()
for i in range(EPOCHS):
    eval_mode()
    
    params_pred = model3(model2(model1(x)))
    y_pred = params_pred[0] * x + params_pred[1]
    print(y_pred.shape, y.shape)
    
    loss = lossfn(y_pred, y)
    loss.backward()
    optimizer1.step()
    optimizer3.step()
    
    optimizer1.zero_grad()
    optimizer3.zero_grad()
       
    test_mode()
    with torch.no_grad():
        x_test = torch.tensor([-10,10], dtype=torch.float)
        x_test = x_test.reshape(-1,1).cuda()
        params = model3(model2(model1(x_test.cuda())))
        y_test = params_pred[0] * x_test + params_pred[1]
        
        y_test = y_test.to('cpu')
        x_test = x_test.to('cpu')
        plt.plot(x_test.numpy(), y_test.numpy(), lw=i*0.01)