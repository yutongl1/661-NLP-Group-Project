import nltk
from nltk import word_tokenize, sent_tokenize,pos_tag
import os
import sys
from nltk.parse import stanford
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.stem import WordNetLemmatizer
import string
import csv
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel

def ansHow(question_tokenized_lower, max_similar_sent):
	question_second = question_tokenized_lower[1]
	question_third = question_tokenized_lower[2]
	temp = ['old', 'long', 'many', 'much', 'tall', 'heavy']
	max_similar_sent_str = " ".join(max_similar_sent)
	answer= ""
	if question_second not in temp:
		answer = max_similar_sent_str
	else:
		tagged = pos_tag(max_similar_sent)
		token_candidates = []
		for token, label in tagged:
			splited = token.split('-')
			if len(splited) > 1:
				for t in splited:
					if t.isdigit():
						token_candidates.append(t)
			if label == 'CD':
				try:
					index = max_similar_sent.index(token)
					if max_similar_sent[index+1] == '%':
						token = token+'%'
					token_candidates.append(token)
				except:
					token_candidates.append(token)
		if len(token_candidates) > 1:
			answer = max_similar_sent_str
		elif len(token_candidates) == 1:
			if question_second == 'long' or (question_second == 'many' and question_third == 'long'):
				try:
					index = max_similar_sent.index(token_candidates[0])
					answer = ' '.join(max_similar_sent[index:index+2])
				except:
					answer = token_candidates[0]  
			else:
				answer = token_candidates[0]  
		else:
			answer = max_similar_sent_str
	#answer = word_tokenize(answer)
	return answer
					

