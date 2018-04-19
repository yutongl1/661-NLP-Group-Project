# Preprocessor for the question generation 
# The preprocessor extracts sentences from .txt wikipedia articles, remove any noises, and tokenize sentences into words. 
import nltk
from nltk import word_tokenize, sent_tokenize,pos_tag
import os
import sys
import re
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
from ansWhy import ansWhy
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



stop_words = set(stopwords.words('english'))


def intersection(lst1, lst2):
	lst3 = set([value for value in lst1 if value in lst2])
	return lst3



def pre_processing(sentences):
	# print "Original:", question
	symbols = {" 's" : "'s", "``":  "''", "$ ": "$", "/ ":"/", " %": "%"}
	sentences = sentences.replace("-lrb- ", "(").replace(" -rrb-", ")").replace("-LRB- ", "(").replace(" -RRB-",")")

	for key in symbols:		
		if key in sentences:
			sentences = sentences.replace(key, symbols[key])				
	# print "Processed:", question
	return sentences


# Tokenize all sentences
def tokenize_sent(article_file):

	sentence_pool = dict()

	article = open(article_file).read()
	article = pre_processing(article)
	paragraphs = [p for p in article.split('\n') if p]

	title = word_tokenize(paragraphs[0].lower())


	for paragraph in paragraphs[1:len(paragraphs)]: # Skip the title
		# =========== Correcting Sentence Tokenization ========= 
		# Idea: Sentence should not end if it contains only one bracket
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

		# p is a list, and each element is a list of sentence_tokens. 
		# p includes sentences for the whole paragraph.  
		p = []
		for sentence in sentences:
			
			sentence = sentence.replace("*", "").lstrip()
			sentence_tokenized = [a.lower() for a in word_tokenize(sentence)]
			p.append(sentence_tokenized)

		# For each sentence, if the paragraph contains only one short sentence, or 
		# if the sentences contains too few wors (<= 4), we exlude it. 
		for sentence_tokenized in p:

			words = [w for w in sentence_tokenized if re.search('[a-zA-Z1-9]', w)]
			if len(words) <= 4 or (len(words) < 10 and len(sentences) <= 1): #or "." not in sentence:
				continue
			sentence_pool[' '.join(sentence_tokenized) ] = p
	# Sentence_pool is a list of strings. 
	return title, sentence_pool


# ====== Cosine Similarity  ======

def cosineSim(sentences_pool, question, question_start, title):

	corpus = [' '.join(question)]


	for s in sentences_pool:
		corpus.append(s)	
		
	# Calculate cosine similarity 
	vec = TfidfVectorizer().fit_transform(corpus)
	
	question_vec = vec[0:1]
	cosine_similarities = linear_kernel(question_vec, vec).flatten()
	# related_docs_indices is the top 8 sentences. 
	related_docs_indices = cosine_similarities.argsort()[:-9:-1]
	
	# Focus on sentences that contain numbers
	if question_start == 'how':
		if question[1] in ['old', 'long', 'many', 'much', 'tall', 'heavy']:
			new_related_indices = [0]
			for idx in related_docs_indices[1:]:
				tags = set([t for w, t in st_pos.tag(corpus[idx].split()) if re.search('[a-zA-Z1-9]', w)])
				if "CD" in tags:
					new_related_indices.append(idx)
			
			if len(new_related_indices) > 1:
				related_docs_indices = new_related_indices

	# Focus on sentences that contain discourse markers for reasons
	if question_start == 'why':
		new_related_indices = [0]
		if cosine_similarities[related_docs_indices[1]] > 0.5:
			new_related_indices.append(related_docs_indices[1])

		for (i, s) in enumerate(corpus):
			for q_word in ['because', 'since', 'so', 'goal', "why", 'reason']:
				if q_word in word_tokenize(s) and i not in new_related_indices:
					new_related_indices.append(i)
					
		if len(new_related_indices) > 1:
			related_docs_indices = new_related_indices


	# for idx in related_docs_indices:
	# 	print "++++ "
	# 	print "sent:", corpus[idx], cosine_similarities[idx]

	confidence = cosine_similarities[related_docs_indices[1]]
	diff_confidence = None
	if len(related_docs_indices) > 2:
		diff_confidence = cosine_similarities[related_docs_indices[1]] - cosine_similarities[related_docs_indices[2]]

	# If the confidence is low, or if the top choices have very close confidence, we lemmatize the candidates and rerank. 
	if confidence < 0.25 or (confidence < 0.55 and diff_confidence and diff_confidence < 0.1):

		max_sentence = None
		max_overlap = None
		max_sentence_idx = None

		tag_dict = {"j":"a", "n":"n", "v": "v"}
		lemmatize_question = [w.replace("-","") for w in corpus[0].split() if w not in stop_words and w not in string.punctuation]
		lemmatize_question = [lemmatizer.lemmatize(w,tag_dict[t[0].lower()]) if t[0].lower() in ['j','n', 'v'] else lemmatizer.lemmatize(w) for w,t in st_pos.tag(lemmatize_question)]			
		
		# print "Lemma Ques:", lemmatize_question
		for idx in related_docs_indices[1:]:
			# print "----"
			# print "sent", corpus[idx]
			lemmatize_sentence = [w.replace("-","") for w in corpus[idx].split() if w not in stop_words and w != "'s"]
			# Augment Year 
			augment_year = []
			for w in lemmatize_sentence:
				if w.isdigit() and len(w) == 4:
					augment_year.append(w[:-1] + '0')
					augment_year.append(w[:-1] + '0s')
					augment_year.append(w[:-2] + '00')
					augment_year.append(w[:-2] + '00s')
			lemmatize_sentence += augment_year
			lemmatize_sentence = [lemmatizer.lemmatize(w,tag_dict[t[0].lower()]) if t[0].lower() in ['j','n', 'v'] else lemmatizer.lemmatize(w) for w,t in st_pos.tag(lemmatize_sentence)]			

			# Discount the focus of the article, which is contained in title. 
			score = 0
			for item in intersection(lemmatize_question,lemmatize_sentence):
				if item not in title:
					score += 1
				else:
					score += 0.2

			if not max_overlap or score > max_overlap:
				max_sentence = corpus[idx]
				max_sentence_idx = idx
				max_overlap = score

			# print "Lemma Sent:", lemmatize_sentence
			# print "intersection: ", intersection(lemmatize_question,lemmatize_sentence), score
		selected_sentence =  max_sentence 
		selected_sentence_idx = max_sentence_idx
		
	else: 
		selected_sentence = corpus[related_docs_indices[1]]
		selected_sentence_idx = related_docs_indices[1]

	return sentences_pool[selected_sentence], word_tokenize(selected_sentence), cosine_similarities[selected_sentence_idx]




 


# TO DO: 
# Questions like: was volta buried where he died or was he buried someplace else

def main():

	# Read in the article and list of questions
	article_file = sys.argv[1]
	question_list = sys.argv[2]

	# Tokenize all sentences
	title, sentences_pool = tokenize_sent(article_file)

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
			max_paragraph, max_similar_sent, max_similarity = cosineSim(sentences_pool, question_tokenized_lower, question_start, title)  


			print "----------"
			print "Question:", question      
			# print "Selected Jaccar:", ' '.join(max_similar_sent1), max_similarity1
			print "Selected Cosine:", ' '.join(max_similar_sent), max_similarity

			# print "======== paragraphs ========"
			# s = ''
			# for s_l in max_paragraph: 
			# 	s += ' '.join(s_l)
			# print s

			# print "======== paragraphs ========"

			# Input lists of tokens for question and max_similar_sentence. 
			# Output a list of tokens
			# if question_start in yes_no_words:
			# 	answer = ansYesNo(question_tokenized_lower, max_similar_sent, max_similarity)
			# 	# print answer
				# print "--------"



			#@ For test
			sent_ = " ".join(max_similar_sent)
			sent_ = list(sent_)
			if sent_:
				sent_[0] = sent_[0].upper()
			sent_ = "".join(sent_)

			if question_start == "when":
				answer = ansWhen(max_similar_sent)
				print(question)
				print(sent_)
				print(answer)
				print("=========")
				
			# if question_start == "where":
			# 	answer = ansWhere(max_similar_sent)
			# 	print(question)
			# 	print(max_similar_sent)
			# 	print(answer)

				

			# if question_start == "how":
			# 	answer = ansHow(question_tokenized_lower, max_similar_sent)
				

			# if question_start == "what":
			# 	answer = ansWhat(parser, question_tokenized_lower, max_similar_sent)
				
				
			# print(question)
			# print(answer)
		

if __name__ == '__main__':
	main()


