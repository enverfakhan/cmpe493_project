#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:26:09 2019

@author: enverfakhan
"""

import os
import re
import sys
import pickle
import string
import random


"""
IO start here
"""
dirname = os.path.dirname(os.getcwd())


def load(flag):
    if flag == 'doc':
        file = 'derlem.txt'
    if flag == 'ques':
        file = 'soru_gruplari.txt'
    path = ('deliverables', file)
    newPath = os.path.join(dirname, *path)
    with open(newPath, 'r', encoding='utf-16') as fh:
        data = fh.read()

    return data


def save(obj, name):
    path = os.path.join(dirname, 'misc', name)
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)



"""
    pre-processing documents
"""


def cleaning(string, EOS=False):
    """
    the input is a possibly conteminated string with wrong encodings, the output
    is cleaned lowercased and the end of sentences is replaced with hashtag
    """

    # before cleaning up, first identify end of the sentences (EOS)
    if EOS:
        pLu = '[{}]'.format("".join([chr(i) for i in range(sys.maxunicode) if chr(i).isupper()]))
        EOS = re.compile(r'([a-z]+|[ş|ı])(\. )((' + pLu + '[a-z]?)|([0-9]+))')
        string = EOS.sub(r'\1#\3', string)

    # period at the end of the sentences are being replaced with hastag (#)
    string = string.lower()
    mapping = {}
    mapping['99_807'] = 231
    mapping['105_770'] = 105
    mapping['117_770'] = 117
    mapping['105_775'] = 105
    mapping['117_776'] = 252
    mapping['115_807'] = 351
    mapping['103_774'] = 287
    mapping['97_770'] = 97
    mapping['111_776'] = 246
    mapping['97_785'] = 97
    Alist = {97, 99, 103, 105, 111, 115, 117}
    solv_prob = []
    flag = False
    for i, c in enumerate(string):
        if flag:
            flag = False
            continue  # pass this character
        if not ord(c) in Alist:
            solv_prob.append(c)  # no need to check this character
        else:
            if i == len(string) - 1:
                continue
            cn = string[i + 1]  # next character
            key = '{}_{}'.format(ord(c), ord(cn))  # creating string with their ordinal
            if key in mapping.keys():  # cheking if this is to be mapped
                solv_prob.append(chr(mapping[key]))  # append the mapped character to the list
                flag = True  # raising flag to pass next character
                continue
            else:
                solv_prob.append(c)

    data = ''.join(solv_prob)
    data = data.replace('iğdır', 'ığdır')
    data = data.replace('irak', 'ırak')
    #    Data= [d if len(d) > 0 else '#' for d in data.splitlines()]   # removing empty lines
    return data


def preprocess(inpString):
    """
    the input is a paragraph string, the output is normalized and splited into
    sentences version of the paragraph
    """

    figure = re.compile(r'\([^\W\d_]+ [0-9]+\.[0-9]+\)')  # i.e (Tablo 3.2)
    out_string = figure.sub('', inpString)

    digit_dot = re.compile(r'([0-9]+)\.([0-9]{3})')  # i.e 19.000 --> 19000
    out_string = digit_dot.sub(r'\1\2', out_string)
    out_string = digit_dot.sub(r'\1\2', out_string)

    centigrade = re.compile(r'(°C)|(°c)|(0C)')  # °C --> santigrat
    out_string = centigrade.sub(r' santigrat', out_string)

    out_string = re.sub(r'°', ' derece', out_string)  # ° --> derece

    digit_space = re.compile(r'([0-9]+) ([0-9]+)')  # 19 000 --> 19000
    out_string = digit_space.sub(r'\1\2', out_string)

    out_string = re.sub(r'â', 'a', out_string)  # Elâzig --> Elazig

    spec_hyphen = re.compile(r'([A-Za-z])-([0-9]+)')  # G-20 --> G20
    out_string = spec_hyphen.sub(r'\1\2', out_string)

    out_string = re.sub(r'-', ' ', out_string)  # replace hyphen with space

    out_string = re.sub(r'%|‰', 'yüzde ', out_string)  # % --> yuzde

    year = re.compile("([0-9]{4})(’|')([a-z]+)")  # 1815'te --> 1815 yilinda
    out_string = year.sub(r'\1 yılında', out_string)

    out_string = re.sub(r' km2', ' kilometrekare', out_string)  # converting km2, m, km
    out_string = re.sub(r' m ', ' metre ', out_string)
    out_string = re.sub(r' km ', ' kilometre ', out_string)

    out_string = re.sub(r"(’|')([a-züşöıç]+)", '', out_string)  # turkiye'de --> turkiye

    out_string = re.sub(r'([0-9]+),([0-9]+)', r'\1CBN\2', out_string)  # replacing comma between
    # digits with a placeholder

    puncs = string.punctuation + '”' + '“' + '’' + '‘'
    translator = str.maketrans('', '', puncs)
    out_string = out_string.translate(translator)  # removing pucntuations

    out_string = re.sub(r'CBN', ',', out_string)  # bringing back the comma between numbers
    # out_string= out_string.split(' ') #[s.split(' ') for s in out_string.split('#')] #splitting from end of sentences
    # end sentence into words

    return out_string


"""
building document and question answer pairs
"""


def doc_builder(clean_string):
    list_of_strings = [l for l in clean_string.splitlines() if len(l) > 0]
    documents = {}
    key = re.compile(r'[0-9]+')
    for istring in list_of_strings:
        index = key.match(istring)
        ind = index.group(0);
        pos = index.end()
        data = preprocess(istring[pos + 1:])
        documents[ind] = data.split(' ')

    return documents


def qa_pairs(string):
    # first split the document into parts ( every part contains question answer pairs
    # and related paragrapgh information
    data = [d + 'EOS' if len(d) > 0 else '#' for d in string.splitlines()]
    data = ''.join(data)
    data = data.split('#')

    # regular expression for question, answer and related paragraph
    q_re = re.compile(r'(s[0-9]+): *(.*\?)');
    a_re = re.compile(r'(c[0-9]+): *(.*)')
    rel_re = re.compile(r'(ilintili paragraf): *([0-9]+)')
    #####
    question_index = {}
    answer_index = {}
    QA_pair = {}
    QP_pair = {}
    for pair in data:
        pair_lines = pair.split('EOS')
        quests = []
        answers = []
        rel_par = []
        for line in pair_lines:
            quests += q_re.findall(line)
            answers += a_re.findall(line)
            rel_par += rel_re.findall(line)
        for quest in quests:
            QA_pair[quest[0]] = answers[0][0]
            if len(rel_par):
                QP_pair[quest[0]] = rel_par[0][1]
            question_index[quest[0]] = preprocess(quest[1]).split(' ')
            answer_index[answers[0][0]] = preprocess(answers[0][1]).split(' ')

    return question_index, answer_index, QA_pair, QP_pair


"""
this is a poor lemmatizer, though it is usefull
"""


def lemmatizer(vocab_index, vocab):

    # find the oov words
    oov = []
    voc = []
    for key in vocab_index.keys():
        try:
            _ = vocab[key]
            voc.append(key)
        except:
            oov.append(key)
            # poorly lemmatize them and look if they are present in the vocab
    lemmatized = []
    doub_exc_new = []
    for word in oov:
        flag = False
        for i in range(len(word)):
            i += 1
            lem = word[:-i]
            if lem in vocab:
                lemmatized.append((word, lem, i))
                flag = True
                break
        if flag is False:
            doub_exc_new.append(word)
    return lemmatized, doub_exc_new, oov, voc


# data selection

def vocab_creator(documents):
    vocab_index = {}
    vocab_freq = {}
    i = 0
    for doc in documents.values():
        for word in doc:
            try:
                vocab_freq[word] += 1
            except:
                vocab_freq[word] = 1
                vocab_index[word] = i
                i += 1

    return vocab_index, vocab_freq


def curating_dataset(parallel_corpus, vocab):
    random.shuffle(parallel_corpus)
    train_en, train_tr = [], []
    new_vocab = set()
    val_tr, val_en = [], []

    for i, tup in enumerate(parallel_corpus):
        t = tup[0]
        e = tup[1]
        words = t.split(' ')

        if len(words) > 30:
            continue

        val = sum([word in vocab for word in words])
        if val > len(words) / 2 and len(val_tr) < 3500:
            val_tr.append(t)
            val_en.append(' '.join(e))
            new_vocab.update(words)
            continue

        if len(new_vocab) < 120000 and val != 0:
            train_tr.append(t)
            train_en.append(' '.join(e))
            new_vocab.update(words)

    return train_tr, train_en, val_tr, val_en


def built_dataset(parallel_corpus, voc_ab):

    new_corpus = curating_dataset(parallel_corpus, voc_ab)
    random.shuffle(new_corpus)
    dev = []
    test = []
    train = []
    for i, tup in enumerate(new_corpus):
        if i < 2000:
            dev.append(tup[0])
            continue
        elif i < 3000:
            test.append(tup[0])
        else:
            train.append(tup[0])

    random.shuffle(dev)
    random.shuffle(train)
    random.shuffle(test)
    dev_tr = [tup[0] for tup in dev]
    dev_en = [tup[1] for tup in dev]
    train_en = [tup[1] for tup in train]
    train_tr = [tup[0] for tup in train]
    test_tr = [tup[0] for tup in test]
    test_en = [tup[1] for tup in test]

    dev_tr = '\n'.join(dev_tr)
    train_tr = '\n'.join(train_tr)
    test_tr = '\n'.join(test_tr)
    dev_en = '\n'.join([' '.join(sentence) for sentence in dev_en])
    train_en = '\n'.join([' '.join(sentence) for sentence in train_en])
    test_en = '\n'.join([' '.join(sentence) for sentence in test_en])

    with open('train.en', 'w') as fh:
        fh.write(train_en)

    with open('train.tr', 'w') as fh:
        fh.write(train_tr)

    with open('dev.tr', 'w') as fh:
        fh.write(dev_tr)

    with open('dev.en', 'w') as fh:
        fh.write(dev_en)

    with open('test.en', 'w') as fh:
        fh.write(test_en)

    with open('test.tr', 'w') as fh:
        fh.write(test_tr)


def obtain_voc(voc):

    data = load('doc')
    strng = load('ques')
    data = cleaning(data)
    strng = cleaning(strng)
    documents = doc_builder(data)
    question_index, answer_index, QA_pair, QP_pair = qa_pairs(strng)
    vocab_index, vocab_freq = vocab_creator(documents)
    vocab_index2, vocab_freq2 = vocab_creator(question_index)
    vocab = set(vocab_index.keys()).union(set(vocab_index2.keys()))
    lemmatized2, doub_exc_new, oov2, voc2 = lemmatizer({s: 0 for s in vocab}, voc)

    return set(voc2)


if __name__ == '__main__':
    data = load('doc')
    strng = load('ques')
    data = cleaning(data)
    strng = cleaning(strng)
    documents = doc_builder(data)
    questions, answers, QA_pair, QP_pair = qa_pairs(strng)




