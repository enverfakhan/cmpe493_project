#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 01:05:28 2019

@author: enverfakhan
"""
from collections import defaultdict
import torch
import random
import numpy as np
from retriever import invertedIndex, retrieve_documents


def laplacian_dist(x, mean, sigma):
    power = -1 * np.sqrt(2) * np.abs(x - mean) / sigma
    cnt = 1 / (np.sqrt(2) * sigma)
    return cnt * np.exp(power)


class Dataset(object):

    def __init__(self, documents, questions, answers, QA_pairs, QD_pairs, word_vec):
        """
        args:
            documents: dict of list of words in docID 
            questions: dict of list of words in questionId
            answeres: dict of list of words in answerID
            QA_pairs: mapping between questionID and answerID
            QD_paris: mapping between questionID and docID
            word_vec: word2vec dictionary; 
                            word_vec['stois']= stois[word]= id
                            word_vec['vect']=  torch.tensor(1, 400)
        """

        self.documents = documents
        self.questions = questions
        self.answers = answers

        self.quest2ans = QA_pairs
        self.quest2doc = QD_pairs

        self.embeddings = word_vec['vectors']
        self.stois = word_vec['stois']

    def _createPoints(self):
        """
        create one point per question, every point holds two container as their 
        attribute:
            dataPoint.docs: 10 document objects which the first five are retrieved
                    documents and the last five are negative sampling for the question 
                    point is built upon. Every document in dataPoint.document is an 
                    object of document class, for detail info refere to 
                    datapoint.document
            dataPoint.quests: 10 question object
            
        """
        self.doc2quest = self._docMapping()

        self.unigram, self.bigram = invertedIndex(self.documents)
        self.points = [dataPoint(key, self) for key in self.questions.keys()]

    def _docMapping(self):
        """
               inverts the quest2doc mapping to doc2quests
            
        """
        doc2quests = defaultdict(list)
        for q, d in self.quest2doc.items():
            doc2quests[d].append(q)
        return doc2quests

    def __getitem__(self, idx):
        return self.points[idx]


class dataPoint(Dataset):

    def __init__(self, questID):
        super(dataPoint, self).__init__()

        self._Docs = self._retrieveDocs(questID)
        self.docs = self._createDocObjects(self._Docs)

        self._Quests = self._retrieveQuestions(questID)
        self.quests = self._createQuestObjects(self._Quests)

        self.the_quest = self.quests[0]

    def _retrieveDocs(self, questID):
        """
            retrieves 10 documents: first document is the related document to the 
            questID, next four documets are the most related documents apart from the 
            true related document, last five documents are randomly picked from 
            documents
            
        """
        question = self.questions[questID]
        the_doc = self.quest2doc[questID]
        tru_docs = retrieve_documents(question, self.unigram, self.bigram, n=5)
        neg_docs = random.sample(self.documents.keys(), 10)

        Docs = [the_doc]
        if the_doc not in tru_docs:
            Docs += tru_docs[:4]
        else:
            tru_docs.remove(the_doc)
            Docs += tru_docs

        for docID in neg_docs:

            if len(Docs) == 10:
                break
            if docID  not in Docs:
                Docs.append(docID)

        return Docs

    def _retrieveQuestions(self, questID):

        """
            retrieves 10 questionIDs, the first one is questID, next four one are 
            related questions of docID in self.Docs[1:5], last five questions are 
            randomly picked from self.questions.keys()
        """
        all_related_quests = set([quest for doc in self._Docs
                                  for quest in self.doc2quest[doc]])
        random_quests = random.sample(self.questions.keys(), 40)

        Quests = [questID]
        Quests += [random.choice(self.doc2quest[dID]) for dID in self._Docs[1:5]]

        for quest in random_quests:

            if len(Quests) == 10:
                break
            if not quest in all_related_quests:
                Quests += quest

        return Quests

    @staticmethod
    def _createDocObjects(DocIDs):
        """
            returns list of docObjects with documents in DocIDs
            
        """
        return [docObject(docId) for docId in DocIDs]

    @staticmethod
    def _createQuestObjects(questIDs):
        """
            returns list of questObjects with questions in questIDs
            
        """
        return [questObject(q) for q in questIDs]


class docObject(Dataset):

    def __init__(self, docID):
        super(docObject, self).__init__()
        self.idx = docID
        self.vector = self._obtainVector()

    def _obtainVector(self):
        ls = [self.embeddings[w] for w in self.documents[self.idx]]
        return torch.stack(ls)

    def answerToQuest(self, questObject):
        length = len(self)
        if not questObject.isRelated(self):
            return torch.tensor([0] * length)
        x = np.zeros(length)
        start, end = self.findPositionOfAnswer(questObject)
        sigma = (end - start)
        mean = (end + start) / 2
        answer = torch.tensor(laplacian_dist(x, mean, sigma))

        return answer

    def findPositionOfAnswer(self, questObject):
        answerId = self.quest2ans[questObject.idx]
        answer = ' '.join(self.answers[answerId])  # string form of the answer list
        document = ' '.join(self.documents[self.id])

        words_pos = [0]  # position of words in the document string
        for i, word in enumerate(self.documents[self.idx]):  # enumerating over words
            pos = words_pos[i] + len(word) + 1
            words_pos.append(pos)
        del words_pos[-1]

        """
            at every begining of the words, commence search for matching, if an unmatch 
            charachter is hit then break the search. When matched increment the value
            of the corresponded word position
            
        """
        chr_match = []
        for w_p_d, c_p_d in enumerate(words_pos):
            chr_match.append(0)
            for c_p_a in range(len(answer)):
                if document[c_p_d + c_p_a] == answer[c_p_a]:
                    chr_match[w_p_d] += 1
                else:
                    break

        start = np.argmax(chr_match)
        end_pos = words_pos[start] + chr_match[start]
        end = np.argmax([i if end_pos - i >= 0 else -1 for i in words_pos])
        print(chr_match)

        return (start, end)

    def __eq__(self, other):
        return self.id == other.id

    def __len__(self):
        return len(self.documents[self.id])


class questObject(Dataset):

    def __init__(self, questID):
        super(questObject, self).__init__()
        self.idx = questID
        self.vector = self._obtainVector()

    def _obtainVector(self):
        ls = [self.embeddings[w] for w in self.questions[self.idx]]
        return torch.stack(ls)

    def isRelated(self, docObject):
        return docObject.idx == self.quest2doc[self.idx]

    @staticmethod
    def similarity(self, word1, word2):
        return self.similarityMatrix[word1][word2]
