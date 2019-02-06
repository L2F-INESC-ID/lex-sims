#!/usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import division
from pyter import ter
from operator import itemgetter
from nltk import word_tokenize
# from nltk.compat import Counter
from nltk.util import ngrams
from collections import Counter
from text_manipulation import to_string
from web2py_utils.search import ncd
import random
import nltk
import re, math
import numpy as np
import os, sys

'''
recebe um vector de palavras e segue pa BINGO
'''
def ter_sim(text, hypo):
	return ter(text, hypo)

def ncd_sim(text, hypo):
	text_string = to_string(text)
	hypo_string = to_string(hypo)

	return ncd(text_string, hypo_string)

def word_matches(h, ref):
	return(len(h & ref))

def precision(h, ref):
	#num matches/len hyp
	num_matches = word_matches(h, ref)
	if len(h) == 0:
		return 0
	else:
		return 1.0*num_matches/len(h)

def recall(h, ref):
	#num matches/len ref
	num_matches = word_matches(h, ref)
	if len(ref):
		return 0
	else:
		return 1.0*num_matches/len(ref)

def meteor(h_precision, h_recall):
	#precision*recall/(1-alpha)*p+alpha*r
	if h_precision == 0 and h_recall == 0:
		return 0
	return 1.0 * (10 * h_precision * h_recall) / (9 * h_precision + h_recall)

def meteor_sim(text, hypo):
	text_c = Counter(text)
	hypo_c = Counter(hypo)

	text_bigram = nltk.bigrams(text)
	hypo_bigram = nltk.bigrams(hypo)
	
	text_trigram = nltk.trigrams(text)
	hypo_trigram = nltk.trigrams(hypo)
	
	text_bigram_c = Counter(text_bigram)
	hypo_bigram_c = Counter(hypo_bigram)
	
	text_trigram_c = Counter(text_trigram)
	hypo_trigram_c = Counter(hypo_trigram)

	text_all_c = text_c + text_bigram_c + text_trigram_c
	hypo_all_c = hypo_c + hypo_bigram_c + hypo_trigram_c

	precision_final = precision(text_all_c, hypo_all_c)
	recall_final = recall(text_all_c, hypo_all_c)

	text_meteor = meteor(precision_final, recall_final)

	return text_meteor

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
	return len(result)

def compute_recall(text, hypo):
	lcs = longest_commnon_subsequence(text, hypo)
	if len(text) == 0:
		return 0
	else:
		return lcs/len(text)

def compute_precision(text, hypo):
	lcs = longest_commnon_subsequence(text, hypo)
	if len(hypo) == 0:
		return 0
	else:
		return lcs/len(hypo)

def skip_recall(text, hypo):
	text_bi_gram = skip_bigrams(text)
	hypo_bi_gram = skip_bigrams(hypo)

	matched_bi_grams = text_bi_gram & hypo_bi_gram

	if len(text_bi_gram) == 0:
		return 0
	else:
		return len(matched_bi_grams) / len(text_bi_gram)

def skip_precision(text, hypo):
	text_bi_gram = skip_bigrams(text)
	hypo_bi_gram = skip_bigrams(hypo)

	matched_bi_grams = text_bi_gram & hypo_bi_gram

	if len(hypo_bi_gram) == 0:
		return 0
	else:
		return len(matched_bi_grams) / len(hypo_bi_gram)

def skip_bigrams(lista, ngram=2):
	vec = []
	bi_gram_set = set()

	for i in range(len(lista)):
		for j in range(len(lista)):
			if(lista[i] != lista[j] and i < j):
				vec.append(itemgetter(i,j)(lista))		
	bi_gram_set.update(vec)
	
	return(bi_gram_set)	

def computeRougeN(text, hypo, ngram=2):
	text_ngrams = set(ngrams(text,ngram))
	hypo_ngrams = set(ngrams(hypo,ngram))

	text_count_ngrams = Counter(text_ngrams)
	hypo_count_ngrams = Counter(hypo_ngrams)

	matched_ngrams = text_ngrams & hypo_ngrams

	if len(hypo_ngrams) == 0:
		return 0

	rouge_N = len(matched_ngrams) / len(hypo_ngrams)

	return rouge_N

def computeRougeL(text, hypo):
	r_lcs = compute_recall(text, hypo)
	p_lcs = compute_precision(text, hypo)

	beta = 1 #p_lcs/r_lcs
	beta_quadrado = beta * beta

	if p_lcs == 0:
		f_score_lcs = 0
	else:
		f_score_lcs =  ((1 + beta_quadrado) * r_lcs * p_lcs ) / (r_lcs + beta_quadrado * p_lcs)
	
	return f_score_lcs

def computeRougeS(text, hypo):
	r_skip = skip_recall(text, hypo)
	p_skip = skip_precision(text, hypo)

	if (r_skip == 0 and p_skip == 0):
		f_skip = 0
	else:
		f_skip = (r_skip * p_skip) / (r_skip + p_skip)

	return f_skip

class TA_SUM(object):
	"""docstring for TA_SUM"""
	def __init__(self, text, hypo):
		self.ter = ter_sim(text, hypo)
		self.ncd = ncd_sim(text, hypo)
		self.met = meteor_sim(text, hypo)
		self.rouge_n = computeRougeN(text, hypo)
		self.rouge_l = computeRougeL(text, hypo)
		self.rouge_s = computeRougeS(text, hypo)
		
