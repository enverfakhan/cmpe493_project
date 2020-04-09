#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:37:11 2019

@author: enverfakhan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 01:05:28 2019

@author: enverfakhan
"""
from collections import defaultdict
import torch
import random
#from random import shuffle
import numpy as np
from retriever import invertedIndex, retrieve_documents

def laplacian_dist(x, mean, sigma):
    power= -1*np.sqrt(2)*np.abs(x-mean)/sigma
    cnt= 1/(np.sqrt(2)*sigma)
    return cnt*np.exp(power)

class Dataset(object):
    
    def  __init__(self, **kargs):
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
        if 's2068' in kargs['questions']:
            del kargs['questions']['s2068']
        self.documents= kargs['documents']
        self.questions= kargs['questions']
        self.answers= kargs['answers']
        
        self.quest2ans= kargs['QA_pairs']
        self.quest2doc= kargs['QD_pairs']
        self.doc2quest= self._docMapping()
        
        self.unigram, self.bigram= invertedIndex(self.documents)
        
        self.embeddings= kargs['word_vec']['vectors']
        self.stois= kargs['word_vec']['stois']
        
        self.points= self._createPoints()
        
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
        
        return [dataPoint(key, self) for key in self.questions.keys()]
        
    
    def _docMapping(self):
        
        """
               inverts the quest2doc mapping to doc2quests
            
        """
        doc2quests= defaultdict(list)
        for q,d in self.quest2doc.items():
            doc2quests[d].append(q)
        return doc2quests
    
  
    def __getitem__(self, idx):
        return self.points[idx]
    
    def __len__(self):
        return len(self.points)
    
class dataPoint(Dataset):
    
    def __init__(self, questID, parent):
#        super(dataPoint, self).__init__()
        self.parent= parent
        self._Docs= self._retrieveDocs(questID)
        self.docs= self._createDocObjects(self._Docs)
        
        self._Quests= self._retrieveQuestions(questID)
        self.quests= self._createQuestObjects(self._Quests)
        
        self.the_quest= self.quests[0]

    
    def _retrieveDocs(self, questID):
        
        """
            retrieves 10 documents: first document is the related document to the 
            questID, next four documets are the most related documents apart from the 
            true related document, last five documents are randomly picked from 
            documents
            
        """
        question= self.parent.questions[questID]
        the_doc= self.parent.quest2doc[questID]
        tru_docs= retrieve_documents(question, self.parent.unigram, self.parent.bigram, n=None)
        neg_docs= random.sample(self.parent.documents.keys(),10)
        tru_docs= [doc[0] for doc in tru_docs]
        
        Docs= [the_doc]
        for doc in tru_docs:
            if doc == the_doc:
                continue
            if len(Docs) == 5:
                break
            if len(self.parent.doc2quest[doc]) == 0:
                continue
            Docs.append(doc)
        if len(Docs) < 5:
            while len(Docs) <5:
                some_arb_docs= random.sample(self.parent.documents.keys(), 20)
                for docID in some_arb_docs:
                    if len(Docs) == 5:
                        break
                    if len(self.parent.doc2quest[docID]) == 0:
                        continue
                    if not docID in Docs:
                        Docs.append(docID)  
                        
        for docID in neg_docs:       
            
            if len(Docs) == 10:
                break            
            if not docID in Docs:
                Docs.append(docID)
                
        return Docs        
        
    
    def _retrieveQuestions(self, questID):
        
        """
            retrieves 10 questionIDs, the first one is questID, next four one are 
            related questions of docID in self.Docs[1:5], last five questions are 
            randomly picked from self.questions.keys()
        """
        all_related_quests= set([quest for doc in self._Docs\
                                 for quest in self.parent.doc2quest[doc] ])
        random_quests= random.sample(self.parent.questions.keys(), 40)
        
        Quests= [questID]
#        print('this is quest id', questID); print(self._Docs)
        for dID in self._Docs[1:5]:
            try:
                ls= random.choice(self.parent.doc2quest[dID])
            except:
                print(questID, dID)
                assert False
            
            Quests.append(ls)
        
        for quest in random_quests:
            
            if len(Quests) == 10:
                break
            if not quest in all_related_quests:
                Quests.append(quest)     
            
        return Quests
    
    def shuffle(self):
        questID= self.the_quest.idx
        self._Docs= self._retrieveDocs(questID)
        self.docs= self._createDocObjects(self._Docs)
        
        self._Quests= self._retrieveQuestions(questID)
        self.quests= self._createQuestObjects(self._Quests)
    
    
    def _createDocObjects(self, DocIDs):
        """
            returns list of docObjects with documents in DocIDs
            
        """
        return [docObject(docId, self.parent) for docId in DocIDs]
    
    
    def _createQuestObjects(self, questIDs):
        """
            returns list of questObjects with questions in questIDs
            
        """
#        print(questIDs)
        return [questObject(q, self.parent) for q in questIDs]


class docObject(Dataset):
    
    def __init__(self, docID, parent):
#        super(docObject, self).__init__()
        self.parent= parent
        self.idx= docID
        self.vector= self._obtainVector()
        
    def _obtainVector(self):
        ls= [self.parent.embeddings[self.parent.stois[w]] for w in self.parent.documents[self.idx]]
        return torch.stack(ls)
    
    def answerToQuest(self, questObject):
        length= len(self)
        if not questObject.isRelated(self):
            return torch.tensor([0]*length)
        x= np.arange(length); start, end= self.findPositionOfAnswer(questObject)
        sigma= (end-start); mean= (end+start)/2
        if sigma == 0:
            sigma= 1/2
        answer= torch.tensor(laplacian_dist(x, mean, sigma))               
        
        return answer
    
    
    def findPositionOfAnswer(self, questObject):
        answerId= self.parent.quest2ans[questObject.idx]
        answer= ' '.join(self.parent.answers[answerId]) # string form of the answer list
        document= ' '.join(self.parent.documents[self.idx])
        
        words_pos=[0]        # position of words in the document string
        for i, word in enumerate(self.parent.documents[self.idx]): # enumerating over words
            pos= words_pos[i] + len(word) +1
            words_pos.append(pos)
        del words_pos[-1]
        
        """
            at every begining of the words, commence search for matching, if an unmatch 
            charachter is hit then break the search. When matched increment the value
            of the corresponded word position
            
        """
        chr_match= []
        for w_p_d, c_p_d in enumerate(words_pos):
            chr_match.append(0)
            for c_p_a in range(len(answer)):
                if c_p_d + c_p_a > len(document) -1:
                    break
                if document[c_p_d + c_p_a] == answer[c_p_a]:
                    chr_match[w_p_d]+=1
                else:
                    break
            
        start = np.argmax(chr_match); end_pos= words_pos[start] + chr_match[start]
        end= np.argmax([i if end_pos - i >=  0 else -1 for i in words_pos])
#        print(chr_match)
        
        return (start, end)
        
       
    
    def __eq__(self, other):
        return self.idx == other.idx
    
    def __len__(self):        
        return len(self.parent.documents[self.idx])
        
class questObject(Dataset):
    
    def __init__(self, questID, parent):
#        super(questObject, self).__init__()
        self.parent= parent
        self.idx= questID
        self.vector= self._obtainVector()
     
    def _obtainVector(self):
#        print(self.idx)
        ls= [self.parent.embeddings[self.parent.stois[w]] for w in self.parent.questions[self.idx]]
        return torch.stack(ls)
        
    def isRelated(self, docObject):
        return docObject.idx == self.parent.quest2doc[self.idx]
    
    def __len__(self):
        return len(self.parent.questions[self.idx])
    
    def __eq__(self, other):
        return self.idx == other.idx
    

            
                 