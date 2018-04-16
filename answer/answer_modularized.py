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


# Tokenize all sentences
def tokenize_sent(article_file):

  sentence_pool = []

  article = open(article_file).read()
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
        # print "======="
        # print "1:", sentences[-1]
        	sentences[-1] = sentences[-1] + sentence
        # print "2:", sentences[-1]
        # print "======="
    	except:
    		sentences.append(sentence)


    for sentence in sentences:
      
      # if sentence[-1] != "." and sentence[-1] != ":" and sentence[-1] != "--":
      #   continue

      # Filter by * : no if with * and too long. 
      sentence = sentence.replace("*", "").lstrip()
      sentence_tokenized = [a.lower() for a in word_tokenize(sentence)]
      words = [w for w in sentence_tokenized if w not in string.punctuation]
      if len(words) <= 4 or (len(words) < 10 and len(sentences) <= 1) or "." not in sentence:
        continue

      # print "------"
      # print ' '.join(sentence_tokenized)
      sentence_pool.append(sentence_tokenized)  
  return sentence_pool



# Calculate the jaccard distance between two strings
# a and b should be a list of tokens in the string
def jaccard_similarity(a,b):
  a = set(a)
  b = set(b)
  c = a.intersection(b)
  return float(len(c)) / (len(a) + len(b) - len(c))


# A Jaccard based similarity measure
def similarity(sent,question_content, percentage):
  #score = len(set(sent).intersection(set(question_content)))
  a = set(sent)
  b = set(question_content)
  c = a.intersection(b)
  score = float(len(c)) / (len(a) + len(b) - len(c))
  # The sentence needs to contain all words in the question content, if not, return 0
  #print(sent)
  for q_word in question_content:
    #fuzzy matching
    found = False
    for s_word in sent:
      percentage_similarity = SequenceMatcher(None, q_word, s_word).ratio()
      if percentage_similarity >= percentage:
        found = True
    if not found:
      score -= 1
      #print(q_word)
  #print
  return  score 


# Customized similarity measure for why type of question
def similarity_why(sent,question_content):
  a = set(sent)
  b = set(question_content)
  c = a.intersection(b)
  score = float(len(c)) / (len(a) + len(b) - len(c))
  
  found = False
  for q_word in ['because','for','since']:
    if q_word not in sent:
      pass
    else:
      found = True
  if not found:
    score = 0
    
  return  score 

# Intersection between two lists
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


# ====== Jaccard Based Similarity ======
def most_similar(sentences_pool, question_tokenized_lower, question_start):

    filtered_list = ['?','when', 'what','where','what','why','which','who','how','do','does','did','a','the','an']
    question_content = [a for a in question_tokenized_lower if a not in filtered_list]  

    # Find the most similar sentences in the pool 
    max_similarity = None
    most_similar_sent = [] #  We need to consider ties

    for sent_idx in range(len(sentences_pool)):
      sent = sentences_pool[sent_idx]
      
      #similarity_score = jaccard_similarity(sent,question_content)+similarity(sent,question_content)
      if question_start == 'why':
        similarity_score = similarity_why(sent,question_content)
      else:
        similarity_score = similarity(sent,question_content, 0.8)
      
      if max_similarity == None:
        max_similarity = similarity_score
        # Append the origin un-lemmatized sentence
        most_similar_sent.append(sentences_pool[sent_idx])
      elif similarity_score > max_similarity:
        max_similarity = similarity_score
        most_similar_sent.append(sentences_pool[sent_idx])
      else:
        pass

    # Now, build the answer from the retrieved sentence
    same_word = set(most_similar_sent[0])
    for s in most_similar_sent[1:]:
      same_word.intersection_update(s)

    # Find the most relevant sentence
    max_similarity_2 = None
    max_similar_sent = None

    for sent in most_similar_sent:
      sent_filtered = [a for a in sent if not a in same_word]
      similarity_socre_2 = similarity(sent_filtered,question_content, 1)
      if max_similarity_2 == None:
        max_similarity_2 = similarity_socre_2
        max_similar_sent = sent
      elif similarity_socre_2 > max_similarity_2:
        max_similarity_2 = similarity_socre_2
        max_similar_sent = sent
    return max_similar_sent, max_similarity_2


# ====== Cosine Similarity  ======

def cosineSim(sentences_pool, question):

  corpus = [' '.join(question)]

  for s in sentences_pool:
    corpus.append(' '.join(s))


  vec = TfidfVectorizer().fit_transform(corpus)
  
  question_vec = vec[0:1]
  cosine_similarities = linear_kernel(question_vec, vec).flatten()
  related_docs_indices = cosine_similarities.argsort()[:-5:-1][1]

  return word_tokenize(corpus[related_docs_indices]), cosine_similarities[related_docs_indices]

 
def yes_no_question(question, max_similar_sent, max_similarity):
	# Yes/No question: answer should contain only yes or no.  

	# print "Question:", ' '.join(question)
	# print "Selected:", ' '.join(max_similar_sent)


	question_parse = parser.parse(question)
	for parse in question_parse:
		verb = parse[0][0].leaves()
		sub = parse[0][1].leaves()
		try:
			obj = parse[0][2].leaves()
		except: 
			obj = sub 

	obj = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in st_pos.tag(obj)]
	max_similar_sent = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in st_pos.tag(max_similar_sent)]

	intersect = intersection(obj,max_similar_sent)
	recall =  float(len(intersect)) / len(obj)
	# print "Score Cosine:", max_similarity
	# print "Score Recall:", recall
	if recall  >= 0.6:
		# Problem: they are not farmed to any extent , but wild kangaroos are shot for meat
		# Asking about the first half and asking about the second half. 
		if "not" not in intersect and "not" in question or "not" in max_similar_sent:
			answer = "No"
		else:
			answer = "Yes"
	else:
		answer = "No"

	# print "Answer: ", answer	 
	# print "-------------"   
	return word_tokenize(answer)

def askWhen(max_similar_sent):
  # 1. Tag: 'DATE', 'TIME'
  # 2.1. one PP or one CD in PP, return it
  # 2.2. multi candidate, return max_similar_sent
  found_DATE = False
  max_similar_sent_tag = ner.tag(max_similar_sent)
  # print max_similar_sent_tag 
  for pair in max_similar_sent_tag:
    if pair[1] == 'DATE' or pair[1] == 'TIME':
      answer = pair[0]
      found_DATE = True
  if not found_DATE:
    #TODO: deal with this situation
    max_similar_parse = parser.parse(max_similar_sent)
    for mparse in max_similar_parse:
      #print mparse
      stack = mparse
      answer = max_similar_parse
      record1 = []                            
      record2 = []
      for i in stack:
        searchLabel(i, "PP", record1)
        # print "-------", record1
      if len(record1) == 1:
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
  return answer

def askWhere(max_similar_sent):
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
  return answer

def ansHow(question_tokenized_lower, max_similar_sent):
  question_second = question_tokenized_lower[1]
  temp = ['old', 'long', 'many', 'much', 'tall', 'heavy']
  max_similar_sent_str = " ".join(max_similar_sent)
  if question_second not in temp:
    answer = max_similar_sent_str
  else:
    number = [int(s) for s in max_similar_sent_str.split() if s.isdigit()]
    tagged = pos_tag(max_similar_sent)
    token_candidates = []
    for token, label in tagged:
      splited = token.split('-')
      if len(splited) > 1:
        for t in splited:
          if t.isdigit():
            token_candidates.append(t)
      if label == 'CD':
        token_candidates.append(token)
    if len(token_candidates) > 1:
      answer = max_similar_sent_str
    elif len(token_candidates) == 1:
      answer = token_candidates[0]  
    else:
      answer = "NULL" 
          

# TO DO: 
# Questions like: was volta buried where he died or was he buried someplace else

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

		for question in f:
			question_tokenized = word_tokenize(question)
			question_tokenized_lower = [a.lower() for a in question_tokenized]
			question_start = question_tokenized_lower[0]

			# max_similar_sent1, max_similarity1  = most_similar(sentences_pool, question_tokenized_lower, question_start)  
			max_similar_sent , max_similarity = cosineSim(sentences_pool, question_tokenized_lower)  


			# print "----------"
			# print "Question:", question      
			# print "Selected Jaccar:", ' '.join(max_similar_sent1), max_similarity1
			# print " "
			# print "Selected Cosine:", ' '.join(max_similar_sent), max_similarity


			# Input lists of tokens for question and max_similar_sentence. 
			# Output a list of tokens
			if question_start in yes_no_words:
				answer = yes_no_question(question_tokenized_lower, max_similar_sent, max_similarity)
				# print answer
				# print "--------"
      
      if question_start == "when":
        answer = ansWhen(max_similar_sent)
        return answer

		  if question_start == "where":
        answer = ansWhere(max_similar_sent)
        return answer

      if question_start == "how":
        answer = ansHow(question_tokenized_lower, max_similar_sent)
        return answer
      	

    

if __name__ == '__main__':
  main()


