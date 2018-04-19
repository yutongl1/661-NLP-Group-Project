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

def ansWhy(question, max_similar_sent, prev_sent):

	reason_words = ['since', 'because']
	answer = max_similar_sent
	for reason_word in reason_words:
		if reason_word in max_similar_sent:
			reason_idx = max_similar_sent.index(reason_word)
			answer = max_similar_sent[reason_idx + 1:]
	if "so" in max_similar_sent:
		so_idx = max_similar_sent.index("so")
		answer = max_similar_sent[:so_idx]
		if answer[-1] in string.punctuation:
			answer = answer[:-1]
	if "this is why" in ' '.join(max_similar_sent) and prev_sent != None:
		answer = prev_sent				
	return answer






