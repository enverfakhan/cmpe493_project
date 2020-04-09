#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:13:48 2019

@author: enverfakhan
"""
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import BCELoss

class customClass(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`. """


    def __init__(self, dimensions):
        super(customClass, self).__init__()

        self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
#        self.linear_in.weight= nn.Parameter(weigths, requires_grad = True)
        self.softmax = nn.Softmax(dim= -1)
    
            
    def forward(self, query, context):

        query_norm = query / query.norm(dim=1)[:, None]
        context_norm = context / context.norm(dim=1)[:, None]
        cos_sim_matrix = torch.mm(query_norm, context_norm.transpose(0,1))
        
        query = self.linear_in(query)
        attention_scores = torch.mm(query, context.t())
        attention_weights = self.softmax(attention_scores)

        stack = torch.stack([torch.dot(attention_weights[i], cos_sim_matrix[i]) for i in range(attention_weights.size(0))])

        y_pred = torch.sum(stack)/attention_weights.size(0)

        return relu(y_pred, inplace= True)


def main(Dataset, model, criterion, optimizer):
    
    for j, data in enumerate(Dataset):
        data.shuffle()
        pred = []
        target = []
        quests = data.quests
        docs = data.docs
        for i, query in enumerate(quests):
            pred.append([])
            target.append([])
            q_vector = query.vector
            for context in docs:
                c_vector = context.vector
                y = torch.tensor([float(query.isRelated(context))], dtype=torch.float)
                y_pred = model(q_vector, c_vector)
                pred[i].append(y_pred)
                target[i].append(y)
        
             
        pred= torch.stack(list(map(torch.stack, pred)))
        target= torch.stack(list(map(torch.stack, target))); target= target.view([10, 10])
        loss= criterion(pred, target)
        if j%21 == 0:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

if __name__ == '__main__':
    loss_history= []
    model= customClass(400)
    criterion= torch.nn.BCELoss(weight=torch.tensor([1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01]))
    optimizer= torch.optim.Adam(model.parameters(), lr= 0.0001)
    for tata in range(10):
        for j, data in enumerate(dataset):
            data.shuffle()
            pred= []; target= []
            quests= data.quests; docs= data.docs
            for i, query in enumerate(quests):
                pred.append([]); target.append([])
                q_vector= query.vector
                for context in docs:
                    c_vector= context.vector
                    y= torch.tensor([float(query.isRelated(context))], dtype= torch.float)
                    y_pred= model(q_vector, c_vector)
                    pred[i].append(y_pred); target[i].append(y)
        
             
            pred = torch.stack(list(map(torch.stack, pred)))
            target = torch.stack(list(map(torch.stack, target))); target= target.view([10, 10])
            loss = criterion(pred, target)
            if j+1 % 20 == 0:
                loss_history.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()