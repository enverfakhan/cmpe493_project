#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:28:03 2019

@author: enverfakhan
"""
import re
import torch
import numpy as np
from basics import preprocess, cleaning
from customLayer import customClass
from retriever import invertedIndex, retrieve_documents
from AnswerModel import customRNNforQuery, customRNNforContext, predictor

raw_doc = 'derlem.txt'


# importers


def obtainRawDoc(raw_doc):
    with open(raw_doc, 'r', encoding='utf-16') as fh:
        doc = fh.read()

    docs = [d for d in doc.splitlines() if len(d) > 0]
    documents = {}
    key = re.compile(r'[0-9]+')
    for istring in docs:
        index = key.match(istring)
        ind = index.group(0);
        pos = index.end()
        data = istring[pos + 1:]
        documents[ind] = data.split(' ')

    return documents


def obtainModels():
    docClassifier = customClass(400)
    queryModel = customRNNforQuery(400, 250)
    contextModel = customRNNforContext(400, 600, 500)
    predictModel = predictor(1000, 500)

    docClassifier.load_state_dict(torch.load('doc.classifier.pt'))
    queryModel.load_state_dict(torch.load('query.model.state.dict._l2_.pt'))
    contextModel.load_state_dict(torch.load('context.model.state.dict._l2_.pt'))
    predictModel.load_state_dict(torch.load('predict.model.state.dict._l2_.pt'))

    return docClassifier, queryModel, contextModel, predictModel


def obtainData():
    structured_data = torch.load('structured.data.pt')

    documents = structured_data['documents']
    unigram, bigram = invertedIndex(documents)

    embeddings = structured_data['word_vec']['vectors']
    stois = structured_data['word_vec']['stois']

    return documents, unigram, bigram, embeddings, stois


def obtainTestData(path):
    with open(path, 'r', encoding='utf-16') as fh:
        rawTest = fh.read()

    cleanTest = cleaning(rawTest)
    preProcessedTest = preprocess(cleanTest)
    questions = [sentence.split() for sentence in preProcessedTest.splitlines()]

    return questions


####################  utility functions         ##############################

def entropy(ls):
    return -sum([p * np.log2(p) for p in ls if p != 0])


def findThePosition(docs):
    positions, confidence = [], []
    for doc in docs:
        temp_doc = np.array([float(d) for d in doc])
        temp_doc = temp_doc / sum(temp_doc)
        cumul_sum = np.cumsum(temp_doc)
        flag = 1
        cnt = 0
        for i, c in enumerate(cumul_sum):

            if flag == 1:
                if c > 0.2:
                    p1 = i
                    flag = 2
            if flag == 2:
                cnt += 1
                if c > 0.8:
                    p2 = i
                    break
                if cnt > len(doc) / 2:
                    p2 = i
                    break
        positions.append((p1, p2))

        entropy_ = entropy(temp_doc);
        mass = float(sum(doc[p1:p2 + 1]))
        confidence.append(mass / (entropy_ + 0000.1))

    return positions, confidence


def decision(positions, confidence, ngram_res, doc_pred):
    scores = [confidence[i] * ngram_res[i][1] * doc_pred[i] for i in range(len(doc_pred))]
    decision = np.argmax(scores)

    t1 = ngram_res[decision][0]

    return t1, positions[decision]


path = '/home/enverfakhan/classess/CMPE493/project/evaluation/test_data/test_questions.txt'

if __name__ == '__main__':

    task1_pred = []
    task2_pred = []
    raw_documents = obtainRawDoc(raw_doc)
    documents, unigram, bigram, embeddings, stois = obtainData()
    docClassifier, queryModel, contextModel, predictModel = obtainModels()
    questions = obtainTestData(path)

    for question in questions:

        vector_quest = []
        for word in question:
            try:
                vector_quest.append(embeddings[stois[word]])
            except:
                vector_quest.append(embeddings[0])

        vector_quest = torch.stack(vector_quest)
        docs = retrieve_documents(question, unigram, bigram)
        docs_vectors = [torch.stack([embeddings[stois[word]] for word in documents[doc[0]]]) for doc in docs]

        doc_pred = np.array([float(docClassifier(vector_quest, d_vector)) for d_vector in docs_vectors])
        doc_pred = doc_pred / sum(doc_pred)
        query_embedd = queryModel(vector_quest.view(len(vector_quest), 1, -1))
        mighty_answers = []
        for i in range(len(docs)):
            doc_embeddings = contextModel(docs_vectors[i])
            predict = [predictModel(query_embedd, ans_embed) for ans_embed in doc_embeddings]
            mighty_answers.append(predict)

        positions, confidence = findThePosition(mighty_answers)
        t1, t2 = decision(positions, confidence, docs, doc_pred)
        task1_pred.append(t1)
        task2_pred.append(raw_documents[t1][t2[0]: t2[1] + 1])
