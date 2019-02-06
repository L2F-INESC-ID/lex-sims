#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

def to_string(s):
	s = str(s)
	string = ' '.join(s).encode('utf-8', 'ignore')
	return string

def remove_ponctuation(text):
	tokenizer = RegexpTokenizer(r'\w+')
	token = tokenizer.tokenize(text)
	return token

def removeStopwords(wordlist):
	stop = stopwords.words("portuguese")
	return [w for w in wordlist if w not in stop]

def stem_text(lista, stem=True):
    if stem:
        stemmer = SnowballStemmer("portuguese")
        tokens = [stemmer.stem(t) for t in lista]
 	#print(tokens)
    return tokens

def split_par(par):
	index = par.index("HYPOTHESIS")
	text = par[1:index]
	hypo = par[(index+1):]
	return(text, hypo)

#Let's define a function to compute what fraction of words in a text are NOT in the stopwords list:
def content_fraction(text):
	stopwords = nltk.corpus.stopwords.words('portuguese')
	content = [w for w in text if w.lower() not in stopwords]
	#print(content)
	return len(content) / len(text)

def chunks(l, n):
	#devolve um vector de strings
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]
