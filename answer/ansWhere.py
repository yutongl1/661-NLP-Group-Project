# Preprocessor for the question generation 
# The preprocessor extracts sentences from .txt wikipedia articles, remove any noises, and tokenize sentences into words. 
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

lemmatizer = WordNetLemmatizer()


# Use stanford pos for lemmatization here 
# The Pos Tagger in NLTK will recognize words like "sits", "walks" as "NNS" instead of verb. 
stanford_pos = '../stanford/stanford-postagger-full-2015-04-20/'
stanford_pos_model = stanford_pos + 'models/english-left3words-distsim.tagger'
stanford_pos_jar = stanford_pos + 'stanford-postagger.jar'
st_pos = StanfordPOSTagger(model_filename=stanford_pos_model, path_to_jar=stanford_pos_jar)


# # NER Tagging:
stanford_ner = '../stanford/stanford-ner-2015-04-20/'
stanford_ner_model = stanford_ner + 'classifiers/english.muc.7class.distsim.crf.ser.gz'
stanford_ner_jar = stanford_ner + 'stanford-ner.jar'
ner = StanfordNERTagger(model_filename=stanford_ner_model, path_to_jar=stanford_ner_jar)


# Set up the stanford PCFG parser
stanford_parser_dir = '../stanford/stanford-parser-full-2015-04-20/'
eng_model_path = stanford_parser_dir  + "englishPCFG.ser.gz"
my_path_to_models_jar = stanford_parser_dir  + "stanford-parser-3.5.2-models.jar"
my_path_to_jar = stanford_parser_dir  + "stanford-parser.jar"
parser=StanfordParser(model_path=eng_model_path, path_to_models_jar=my_path_to_models_jar, path_to_jar=my_path_to_jar)

stopWords = stopwords.words('english')

# cur: currect tree
# label: target label
# record: candidates
def searchLabel(cur, label, record):
	answer = None
	if cur.label() == label:
		# record.append(cur.leaves())
		record.append(cur)
	for i in cur:
		# print "--",    (i), isinstance(i, (str, unicode)), i
		if not isinstance(i, (str, unicode)) and i.label() == label:
			# record.append(i.leaves())
			record.append(i)
		else:
			if len(i):
				if isinstance(i[0], (str, unicode)):
					continue
				else:
					for j in i:
						searchLabel(j, label, record)

def ansWhere(max_similar_sent):
	# 1. Tag: 'DATE', 'TIME'
	found_LOCATION = False
	max_similar_sent_tag = ner.tag(max_similar_sent)
	for pair in max_similar_sent_tag:
		if pair[1] == 'LOCATION' or pair[1] == 'LOCATION':
			answer = pair[0]
			found_LOCATION = True
	if not found_LOCATION:
		max_similar_parse = parser.parse(max_similar_sent)
		for mparse in max_similar_parse:
			#print mparse
			stack = mparse
			answer = max_similar_sent
			record1 = []
			record2 = []
			for i in stack:
				searchLabel(i, "PP", record1)
				# print "-------", record1
			if len(record1) == 1:
				if record1[0][0][0] in ("in", "from", "at", "on", "under"):
					answer = record1[0].leaves()     
			else:
				for j in record1:
					searchLabel(j, "CD", record2)
				if len(record2) == 1:
							answer = record2[0].leaves()
	answer = " ".join(answer)
	a = list(answer)
	if a:
		a[0] = a[0].upper()
	answer = "".join(a)
	print answer
	return answer