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
from ansWhat import ansWhat
from ansWhere import ansWhere
from ansWhen import ansWhen
from ansHow import ansHow
from ansYesNo import ansYesNo
from ansYesNo import intersection


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


def pre_processing(sentences):
	# print "Original:", question
	symbols = {" 's" : "'s", " ," : ",", "`` ":  "''", " ''": "''", " :" : ":", "  ": " ", "$ ": "$", "/ ":"/"}
	sentences = sentences.replace("-lrb- ", "(").replace(" -rrb-", ")").replace("-LRB- ", "(").replace(" -RRB-",")")

	for key in symbols:		
		if key in sentences:
			sentences = sentences.replace(key, symbols[key])				
	# print "Processed:", question
	return sentences


# Tokenize all sentences
def tokenize_sent(article_file):

	sentence_pool = []

	article = open(article_file).read()
	article = pre_processing(article)
	paragraphs = [p for p in article.split('\n') if p]
	for paragraph in paragraphs[1:len(paragraphs)]: # Skip the title

		sentences_rough = sent_tokenize(paragraph.decode('utf-8'))

		sentences = []

		symbol_dict = {"]":"[", "}":"{", ")":"(", '"' : '"'}
		for sentence in sentences_rough:
			symbol_stack = []
			add_sentence = True
			for c in sentence:
				if c in symbol_dict.values():
					if c == '"' and symbol_stack and symbol_stack[-1] == '"':
						symbol_stack.pop()
					else:
						symbol_stack.append(c)
				
				elif c in symbol_dict.keys():
					if symbol_stack == [] or symbol_dict[c] != symbol_stack.pop():
						add_sentence = False

			if add_sentence:
				sentences.append(sentence)
			else:
				try:
					sentences[-1] = sentences[-1] + sentence
				except:
					sentences.append(sentence)


		for sentence in sentences:
			
			sentence = sentence.replace("*", "").lstrip()
			sentence_tokenized = [a.lower() for a in word_tokenize(sentence)]
			words = [w for w in sentence_tokenized if w not in string.punctuation]
			if len(words) <= 4 or (len(words) < 10 and len(sentences) <= 1) or "." not in sentence:
				continue

			# print "------"
			# print ' '.join(sentence_tokenized)
			sentence_pool.append(sentence_tokenized)  
	return sentence_pool


# ====== Cosine Similarity  ======

def cosineSim(sentences_pool, question, question_start):

	corpus = [' '.join(question)]

	if question_start == 'why':
		for s in sentences_pool:
			for q_word in ['because', 'since', 'so', 'goal']:
				if q_word in s:
					corpus.append(' '.join(s))
	else:
		for s in sentences_pool:
			corpus.append(' '.join(s))

	vec = TfidfVectorizer().fit_transform(corpus)
	
	question_vec = vec[0:1]
	cosine_similarities = linear_kernel(question_vec, vec).flatten()
	related_docs_indices = cosine_similarities.argsort()[:-5:-1][1]

	return word_tokenize(corpus[related_docs_indices]), cosine_similarities[related_docs_indices]


def main():

	# Read in the article and list of questions
	article_file = sys.argv[1]
	question_list = sys.argv[2]

	# Tokenize all sentences
	sentences_pool = tokenize_sent(article_file)

	yes_no_words = ["is","was","are","were","do","does","did","have","has","had"]
	yes_no_words += map(lambda w : w + "n't", yes_no_words)      

	
	with open(question_list) as f:
	# For each question on the list
		count = 0
		for question in f:
			question = pre_processing(question)

			question_tokenized = word_tokenize(question)
			question_tokenized_lower = [a.lower() for a in question_tokenized]
			question_start = question_tokenized_lower[0]

			# max_similar_sent1, max_similarity1  = most_similar(sentences_pool, question_tokenized_lower, question_start)  
			max_similar_sent , max_similarity = cosineSim(sentences_pool, question_tokenized_lower, question_start)  


			# print "----------"
			# print "Question:", question      
			# print "Selected Jaccar:", ' '.join(max_similar_sent1), max_similarity1
			# print "Selected Cosine:", ' '.join(max_similar_sent), max_similarity


			# Input lists of tokens for question and max_similar_sentence. 
			# Output a list of tokens
			if question_start in yes_no_words:
				answer = ansYesNo(question_tokenized_lower, max_similar_sent, max_similarity)
				# print answer
				# print "--------"
			
			if question_start == "when":
				answer = ansWhen(max_similar_sent)
				

			if question_start == "where":
				answer = ansWhere(max_similar_sent)
				

			if question_start == "how":
				answer = ansHow(question_tokenized_lower, max_similar_sent)
				

			if question_start == "what":
				answer = ansWhat(parser, question_tokenized_lower, max_similar_sent)
				
				
			print(question)
			print(answer)
		

if __name__ == '__main__':
	main()


