#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import re
from string_features import *

def get_numbers(text):
	numbers = []
	for token in text:
		if re.match(r'\d', token):
			numbers.append(token)
	return numbers

def get_not_numbers(text):
	not_numbers = []
	for token in text:
		if re.match(r'\D', token):
			not_numbers.append(token)
	return not_numbers

def index(numero, lista):
	index = lista.index(numero)
	return index

def pad_list(lista, pad):
	lista.reverse()
	lista += [None] * pad
	lista.reverse()
	lista += [None] * pad
	return lista

def get_context(index, lista, pad):
	if(lista[0] is not None):
		lista = pad_list(lista, pad)
	nova_lista = []
	for x in range(-pad, pad+1):
		if lista[index+(x+pad)] is not None:
			nova_lista.append(lista[index+(x+pad)])
	return nova_lista

def get_number(vector):
	for token in vector:
		if re.match(r'\d', token):
			return token

def merge_list(vector_de_listas):
	tamanho = len(vector_de_listas)
	contador = 0
	merged_list = []
	while (contador < tamanho):
		merged_list += vector_de_listas[contador]
		contador += 1
	return list(set(merged_list))

def calcula_sim(text, hypo, pad=2):
		text_digits = get_numbers(text)
		hypo_digits = get_numbers(hypo)
		if not text_digits or not hypo_digits:
			return 0

		text_output_list = []
		hypo_output_list = []

		digits_index_text_list = []
		digits_index_hypo_list = []

		for i in range(len(text_digits)):
			digits_index_text_list.append(index(text_digits[i], text))	
		for j in range(len(hypo_digits)):
			digits_index_hypo_list.append(index(hypo_digits[j], hypo))

		for k in range(len(digits_index_text_list)):
			text_output_list.append(get_context(digits_index_text_list[k], text, pad))

		for l in range(len(digits_index_hypo_list)):
			hypo_output_list.append(get_context(digits_index_hypo_list[l], hypo, pad))

		
		text_output = merge_list(text_output_list)
		hypo_output = merge_list(hypo_output_list)
		
		digit_sim = 1.0 - jaccard_sim(text_digits, hypo_digits)
		
		text_not_numbers = get_not_numbers(text_output)
		hypo_not_numbers = get_not_numbers(hypo_output)
		
		non_digit_sim = jaccard_sim(text_not_numbers, hypo_not_numbers)
		
		
		total_sim = digit_sim * non_digit_sim
		
		return total_sim


class Numeric_features(object):
	def __init__(self, text, hypo):
		self.num = calcula_sim(text, hypo)

	