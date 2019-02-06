#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
from string_features import jaccard_sim

def lemmatize(word):
    stemmer = nltk.stem.snowball.PortugueseStemmer()
    lemma = stemmer.stem(word)
    if lemma is not None:
        return lemma
    return word

def ne(token):
	#This just assumes that words in all caps or titles are named entities.
	if token.istitle() or token.isupper():
		return True
	return False

class RTE_features(object):

	def __init__(self, text, hypothesis,lemmatize=False):
		self.negwords = set(['não', 'nao', 'nunca', 'jamais', 'nada', 'nenhum', 'ninguém'])	#'not', 'no', 'never', 'nothing', 'none', 'nobody'
		#'can', 'could', 'may', 'might', 'will', 'would', 'must', 'shall' 'should', 'possible', 'possibly'
		self.modalwords = set(['podia','poderia','dever','deve','devia','deverá','deveria','faria','possivel','possibilidade','possa'])
		self.text_words = set(text)
		self.hypothesis_words = set(hypothesis)

		if lemmatize:
			self.text_words = set(lemmatize(token) for token in self.text_words)
			self.hypothesis_words = set(lemmatize(token) for token in self.hypothesis_words)

		self._overlap = (self.text_words & self.hypothesis_words)
		self._text_extra = jaccard_sim(self.text_words,self.hypothesis_words)
		self._hypothesis_extra = self.hypothesis_words - self.text_words

		self._racio_neg = jaccard_sim( (self.negwords & self.text_words),(self.negwords & self.hypothesis_words))
		self._non_common = jaccard_sim(self.text_words,self.hypothesis_words)
		self._racio_modal = jaccard_sim( (self.modalwords & self.text_words) , (self.modalwords & self.hypothesis_words))

	def overlap(self, token_type, debug=False):
		ne_in_text = set(token for token in self.text_words if ne(token))
		ne_in_hypo = set(token for token in self.hypothesis_words if ne(token))
		ne_overlap = jaccard_sim(ne_in_text, ne_in_hypo)
		if token_type == 'ne':
			if debug:
				print("ne_overlap", ne_overlap)
			return ne_overlap
		elif token_type == 'word':
			if debug:
				print("word_overlap", self._overlap - ne_overlap)
			return self._overlap - ne_overlap
		else:
			raise ValueError("Type not recognized:'%s'" % token_type)
