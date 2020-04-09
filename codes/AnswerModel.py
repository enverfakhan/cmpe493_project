#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:27:55 2019

@author: enverfakhan
"""
import torch
import torch.nn as nn


# from torch.nn import Variable
# import customLayer

class customRNNforContext(nn.Module):

    def __init__(self, inp_dim, hid_dim, out_dim):
        super(customRNNforContext, self).__init__()

        self.hidden_size = hid_dim

        self.i2h = nn.Linear(inp_dim + hid_dim, hid_dim)
        self.i2o = nn.Linear(inp_dim + hid_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, input_sentence):
        hidden = self.initHidden()
        outputs = []
        for vector in input_sentence.view(len(input_sentence), 1, -1):
            combined = torch.cat((vector, hidden), 1)
            hidden = self.i2h(combined)
            output = self.i2o(combined)
            outputs.append(self.tanh(output))
        return outputs

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class customRNNforQuery(nn.Module):

    def __init__(self, inp_dim, hidden_dim):
        super(customRNNforQuery, self).__init__()
        self.LSTM = nn.LSTM(inp_dim, hidden_dim, bidirectional=True)

    def forward(self, inp_sentence):
        output, hidden = self.LSTM(inp_sentence)
        return output[-1]


class predictor(nn.Module):

    def __init__(self, inp_dim, hidden_dim):
        super(predictor, self).__init__()
        self.linear_in = nn.Linear(inp_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, context_word):
        combined = torch.cat((query, context_word), 1)
        hidden = self.linear_in(combined)
        hidden = self.tanh(hidden)
        output = self.linear_out(hidden)
        pred = self.sigmoid(output)
        return pred


if __name__ == '__main__':

    queryModel = customRNNforQuery(400, 250)
    contextModel = customRNNforContext(400, 600, 500)
    predictModel = predictor(1000, 500)

    queryModel.load_state_dict(torch.load('query.model.state.dict._l_.pt'))
    contextModel.load_state_dict(torch.load('context.model.state.dict._l_.pt'))
    predictModel.load_state_dict(torch.load('predict.model.state.dict._l_.pt'))

    criterion = torch.nn.modules.loss.KLDivLoss(reduction='sum')

    query_optim = torch.optim.SGD(queryModel.parameters(), lr=0.00001)
    context_optim = torch.optim.SGD(contextModel.parameters(), lr=0.00001)
    prediction_optim = torch.optim.SGD(predictModel.parameters(), lr=0.00001)
    flag = True
    loss_history = []
    for j in range(2):
        if flag:

            for i, data in enumerate(dataset[:-20]):

                queryModel.zero_grad()
                contextModel.zero_grad()
                predictModel.zero_grad()

                query = data.the_quest.vector
                doc = data.docs[0]
                doc_vector = doc.vector

                query_tensor = query.view(len(query), 1, 400)

                answer = doc.answerToQuest(data.the_quest).type(torch.float32)

                query_embedd = queryModel(query_tensor)
                ans_embeddings = contextModel(doc_vector)
                pred = [predictModel(query_embedd, ans_embed) for ans_embed in ans_embeddings]
                pred = torch.stack(pred).view_as(answer)
                res = (sum(answer) / sum(pred)) * pred

                loss = criterion(torch.log(res), answer)
                if i % 40 == 0:
                    print('{}th datapoint {}th step'.format(i, j))
                    print(loss)
                    loss_history.append(loss)
                if loss != loss:
                    print(' report nan gradient at training step {} example {}'.format(j, i))
                    torch.save(queryModel.state_dict(), 'query.model.state.dict._{}_.pt'.format(j))
                    torch.save(contextModel.state_dict(), 'context.model.state.dict._{}_.pt'.format(j))
                    torch.save(predictModel.state_dict(), 'predict.model.state.dict._{}_.pt'.format(j))
                    queryModel.zero_grad();
                    contextModel.zero_grad();
                    predictModel.zero_grad()
                    continue

                loss.backward()
                query_optim.step()
                context_optim.step()
                prediction_optim.step()

    torch.save(queryModel.state_dict(), 'query.model.state.dict._l2_.pt'.format(j))
    torch.save(contextModel.state_dict(), 'context.model.state.dict._l2_.pt'.format(j))
    torch.save(predictModel.state_dict(), 'predict.model.state.dict._l2_.pt'.format(j))
