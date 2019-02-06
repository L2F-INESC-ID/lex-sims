#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import random
import nltk
import re, math
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from text_manipulation import to_string
import numpy as np
import os
import sys, pickle
# sys.path.insert(0, '/home/ricardo/Downloads/jaro_winkler-1.0.2/jaro/')
import jaro

def bigger_string(one, two):
    one_string = ''.join(one)
    two_string = ''.join(two)
    if len(one_string) > len(two_string):
        return len(one_string)
    else:
        return len(two_string)

def bigger_pair(pair_one, pair_two):
    if len(pair_one) > len(pair_two):
        return len(pair_one)
    else:
        return len(pair_two)

WORD = re.compile(r'\w+')

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

def min_max(label1, label2):
    if len(label1) > len(label2):
        minimo = len(label2)
        maximo = len(label1)
    else:
        minimo = len(label1)
        maximo = len(label2)

    return (minimo, maximo)

def jaccard_sim(seq1, seq2):
    """Compute the jaccard_similarity_score between the two sequences `seq1` and `seq2`.
    They should contain hashable items.

    The return value is a float between 0 and 1, where 1 means equal, and 0 totally different.
    """
    set1, set2 = set(seq1), set(seq2)

    if float(len(set1 | set2)) == 0.0:
        return 0.0
    return len(set1 & set2) / float(len(set1 | set2))

def levenshteinDistance(s1,s2):
        if len(s1) > len(s2):
            s1,s2 = s2,s1
        distances = range(len(s1) + 1)
        for index2,char2 in enumerate(s2):
            newDistances = [index2+1]
            for index1,char1 in enumerate(s1):
                if char1 == char2:
                    newDistances.append(distances[index1])
                else:
                    newDistances.append(1 + min((distances[index1],
                                                 distances[index1+1],
                                                 newDistances[-1])))
            distances = newDistances
        return distances[-1]

def longest_commnon_subsequence(a, b):
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = \
                    max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = a[x-1] + result
            x -= 1
            y -= 1
    return len(result)/bigger_pair(a,b)

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def normalized_len(um, dois):
    if len(um) > len(dois):
        return len(dois)/len(um)
    else:
        return len(um)/len(dois)

def remove(par):
    par = par.replace("TEXT: ", "")
    par = par.replace("HYPOTHESIS: ", "")
    return par

tf = TfidfVectorizer(lowercase=False,tokenizer=None, stop_words=None)

def soft_tfidf(text, hypo):
    global tf
    #print(text)
    text = ' '.join(text)
    hypo = ' '.join(hypo)
    #print(text)
    text_resp = tf.transform([text])
    hypo_resp = tf.transform([hypo])

    valores_tf_idf_text = []
    valores_tf_idf_hypo = []

    features_names_text = []
    features_names_hypo = []

    feature_names = tf.get_feature_names()
    for col in text_resp.nonzero()[1]:
        #print feature_names[col], ' - ', text_resp[0, col]
        features_names_text.append(feature_names[col])
        valores_tf_idf_text.append(text_resp[0,col])
    #print("---------------")
    for col in hypo_resp.nonzero()[1]:
        #print feature_names[col], ' - ', hypo_resp[0, col]
        features_names_hypo.append(feature_names[col])
        valores_tf_idf_hypo.append(hypo_resp[0,col])

    valor = 0
    for i, word_t in enumerate(features_names_text):
        for j, word_h in enumerate(features_names_hypo):
            if jaro.jaro_winkler_metric(word_t, word_h) > 0.9:
                valor += jaro.jaro_winkler_metric((word_t), (word_h)) * valores_tf_idf_text[i] * valores_tf_idf_hypo[j]
    #print(valor)
    return valor


# !!!! TODO / check on MSRP
def cria_tf_idf(data, clusterfilename=''):
    global tf

    if clusterfilename and os.path.isfile("cacheric/" + clusterfilename):
        print "Cached tf-idf: cacheric/" + clusterfilename

        with open("cacheric/" + clusterfilename, 'rb') as f1:
            tf = pickle.load(f1)

        return

    print "building tf-idf..."
    doc = [remove(d).replace('\n', ' ') for d in data]
    tfs = tf.fit_transform(doc)

    if clusterfilename and not os.path.isfile("cacheric/" + clusterfilename):
        print "caching tf-idf into: cacheric/" + clusterfilename

        if not os.path.exists("cacheric"):
            os.makedirs("cacheric")

        with open("cacheric/" + clusterfilename, 'wb') as f1:
            pickle.dump(tf, f1)

class String_features(object):

    def __init__(self, text, hypo):
        self.edit_distance = levenshteinDistance(text, hypo)
        self.lcs = longest_commnon_subsequence(text, hypo)
        self.len = normalized_len(text, hypo)
        self.cosine = get_cosine(text_to_vector(to_string(text)), text_to_vector(to_string(hypo)))
        mini, maxi = min_max(text, hypo)
        self.minimo = mini
        self.maximo = maxi
        self.jaccard = jaccard_sim(text, hypo)
        self.tfidf = soft_tfidf(text, hypo)


