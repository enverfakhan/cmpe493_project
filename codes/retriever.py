#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:29:08 2019

@author: enverfakhan
"""

from collections import defaultdict
from collections import Counter

def invertedIndex(documents):
    
    unigram_invertedIndex = defaultdict(set)
    bigram_invertedIndex = defaultdict(set)
    for ind, doc in documents.items():
        l = len(doc)-1
        for i, word in enumerate(doc):
            unigram_invertedIndex[word].add(ind)
            if i < l:
                bigram = '{} {}'.format(doc[i], doc[i+1])
                bigram_invertedIndex[bigram].add(ind)
    
    return unigram_invertedIndex, bigram_invertedIndex


def retrieve_documents(question, unigram, bigram, n=10):
    l = len(question)-1
    bi_q = []
    for i, q in enumerate(question):
        if i < l:
            bi = '{} {}'.format(q, question[i+1])
            bi_q.append(bi)
        
    docs = Counter()
    [docs.update(bigram[bi]) for bi in bi_q]
    [docs.update(unigram[u]) for u in question]

    return docs.most_common(n)