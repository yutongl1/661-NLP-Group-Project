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
import re

lemmatizer = WordNetLemmatizer()
stop_words = set([u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'o', u'hadn', u'herself', u'll', u'had', u'should', u'to', u'only', u'won', u'under', u'ours', u'has', u'do', u'them', u'his', u'very', u'they', u'during', u'now', u'him', u'nor', u'd', u'did', u'didn', u'this', u'she', u'each', u'further', u'where', u'few', u'because', u'doing', u'some', u'hasn', u'are', u'our', u'ourselves', u'out', u'what', u'for', u'while', u're', u'does', u'above', u'between', u'mustn', u't', u'be', u'we', u'who', u'were', u'here', u'shouldn', u'hers', u'by', u'on', u'about', u'couldn', u'of', u'against', u's', u'isn', u'or', u'own', u'into', u'yourself', u'down', u'mightn', u'wasn', u'your', u'from', u'her', u'their', u'aren', u'there', u'been', u'whom', u'too', u'wouldn', u'themselves', u'weren', u'was', u'until', u'more', u'himself', u'that', u'but', u'don', u'with', u'than', u'those', u'he', u'me', u'myself', u'ma', u'these', u'up', u'will', u'below', u'ain', u'can', u'theirs', u'my', u'and', u've', u'then', u'is', u'am', u'it', u'doesn', u'an', u'as', u'itself', u'at', u'have', u'in', u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other', u'which', u'you', u'shan', u'needn', u'haven', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'm', u'yours', u'so', u'y', u'the', u'having', u'once'])


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
parser = StanfordParser(model_path=eng_model_path, path_to_models_jar=my_path_to_models_jar, path_to_jar=my_path_to_jar)

stopWords = stopwords.words('english')

negation_words = ['not', 'few']


# Intersection between two lists
def intersection(lst1, lst2):
		lst3 = [value for value in lst1 if value in lst2]
		return lst3


def ansYesNo(question, max_similar_sent, max_similarity, title):
	# Yes/No question: answer should contain only yes or no.  

	if question[-1] == "?":
		question = question[:-1]
	question_pos = st_pos.tag(question)
	max_similar_pos = st_pos.tag(max_similar_sent)

	question_parse = parser.parse(question)
	for parse in question_parse:
		verb = parse[0][0].leaves()
		sub = None 
		obj = None
		try:
			sub = parse[0][1].leaves()
			obj = parse[0][2].leaves()
			obj = [w for w in question if w not in verb and w not in sub]
		except: 
			obj = sub 

		if not obj:
			obj = sub
			if not obj:
				obj = question

	# Remove stopwords and punctuations
	obj = [w for w in obj if not w in stop_words and not w in string.punctuation]
	selected = [w for w in max_similar_sent if not w in stop_words]

	if len(obj) <= 2 and len(intersection(obj,selected)) > 0: 
		obj = [w for w in question[1:] if not w in stop_words and not w in string.punctuation]



	# Lemmatize
	tag_dict = {"j":"a", "n":"n", "v": "v"}
	obj = [lemmatizer.lemmatize(w,tag_dict[t[0].lower()]) if t[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(w) for w,t in question_pos if w in obj]
	obj = [w for w in obj if re.search('[a-zA-Z1-9]', w) and w != "'s"]
	selected = [lemmatizer.lemmatize(w,tag_dict[t[0].lower()]) if t[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(w) for w,t in max_similar_pos if w in selected]
	selected = [w for w in selected if re.search('[a-zA-Z1-9]', w) and w != "'s"]
	# Calculate how many overlapped words is in the question. 
	intersect = intersection(obj,selected)

	obj_score = 0 
	intersect_score = 0
	for obj_item in obj:
		if obj_item in title: 
			obj_score += 0.2
		else:
			obj_score += 1
	for intersect_item in intersect:
		if intersect_item in title:
			intersect_score += 0.2
		else:
			intersect_score += 1

	recall =  float(intersect_score) / obj_score

	# print "--------"
	# print 'obj', obj
	# print "Intersect:", intersect
	# print "Score Recall:", recall
	# print "--------"

	if recall >= 0.5 and 'or' in question: 
			answer = ' '.join(max_similar_sent)		

	elif recall  > 0.6:

		answer = "Yes"
		negation = False
		in_question = 0
		in_sentence = 0


		for neg_words in negation_words:
			if neg_words in question:
				in_question += 1
			if neg_words in max_similar_sent:
				in_sentence += 1
			if abs(in_question - in_sentence) % 2 != 0:
				negation = True

		for neg_words in negation_words:
			if neg_words in question and negation:
				try:
					after_not, tag = question_pos[question.index(neg_words) + 1]
				except:
					continue

				if tag[0].lower() in ['j','n','v']:
					after_not = lemmatizer.lemmatize(after_not,tag_dict[tag[0].lower()])
				if after_not in intersect:
					answer = "No"

			if neg_words in max_similar_sent and negation:
				try:
					after_not, tag = max_similar_pos[max_similar_sent.index(neg_words) + 1]
				except:
					continue

				if tag[0].lower() in ['j','n','v']:
					after_not = lemmatizer.lemmatize(after_not,tag_dict[tag[0].lower()])
				if after_not in intersect:
					answer = "No"		


		for qw in question:
			if qw.isdigit():
				number_match = False
				almost_match = False
				for w in max_similar_sent:
					if qw == w:
						number_match = True
						continue
					elif w.isdigit() and len(w) == len(qw):
						almost_match = True
				if not number_match and almost_match:
					if negation:
						answer = "Yes"
					else: 
						answer = "No"
	else:
		answer = "No"

	# print "Answer: ", answer	 
	# print "-------------"   
	return word_tokenize(answer)