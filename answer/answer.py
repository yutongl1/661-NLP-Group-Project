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

lemmatizer = WordNetLemmatizer()



#	Calculate the jaccard distance between two strings
#	a and b should be a list of tokens in the string
def jaccard_similarity(a,b):
	a = set(a)
	b = set(b)
	c = a.intersection(b)
	return float(len(c)) / (len(a) + len(b) - len(c))

#	A Jaccard based similarity measure
def similarity(sent,question_content, percentage):
	#score = len(set(sent).intersection(set(question_content)))
	a = set(sent)
	b = set(question_content)
	c = a.intersection(b)
	score = float(len(c)) / (len(a) + len(b) - len(c))
	#	The sentence needs to contain all words in the question content, if not, return 0
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
	return	score 


#	Customized similarity measure for why type of question
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
		
	return	score 

#	Intersection between two lists
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def generate_bigram(tokens):
	return zip(tokens, tokens[1:])

def main():
	# stanford_pos_dir = '/Users/yuyanzhang/Desktop/CMU/NLP/project/tools/stanford-postagger-full-2015-04-20/'
	# eng_model_filename= stanford_pos_dir + 'models/english-bidirectional-distsim.tagger'
	# my_path_to_jar= stanford_pos_dir + 'stanford-postagger.jar'
	# st = StanfordPOSTagger(model_filename=eng_model_filename, path_to_jar=my_path_to_jar) 
	# print(st.tag('What is the airspeed of an unladen swallow ?'.split()))


	# # NER Tagging:
	stanford_ner = '/Users/wen/Education/2018-Spring-CMU/NLP/661-NLP-Group-Project-master/stanford/stanford-ner-2015-04-20/'
	# stanford_ner_model = stanford_ner + 'classifiers/english.all.3class.distsim.crf.ser.gz'
	stanford_ner_model = stanford_ner + 'classifiers/english.muc.7class.distsim.crf.ser.gz'
	stanford_ner_jar = stanford_ner + 'stanford-ner.jar'
	ner = StanfordNERTagger(model_filename=stanford_ner_model, path_to_jar=stanford_ner_jar)
	#print(ner.tag('Rami Eid is studying at Stony Brook University in NY'.split()))

	# Set up the stanford PCFG parser
	stanford_parser_dir = '/Users/wen/Education/2018-Spring-CMU/NLP/661-NLP-Group-Project-master/stanford/stanford-parser-full-2015-04-20/'
	eng_model_path = stanford_parser_dir  + "stanford-parser-3.5.2-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
	my_path_to_models_jar = stanford_parser_dir  + "stanford-parser-3.5.2-models.jar"
	my_path_to_jar = stanford_parser_dir  + "stanford-parser.jar"
	parser=StanfordParser(model_path=eng_model_path, path_to_models_jar=my_path_to_models_jar, path_to_jar=my_path_to_jar)
	# sent = "Seth Kramer, one of the directors, describes how he first got the idea for The Linguists when in Vilnius, Lithuania, he could not read Yiddish inscriptions on a path in spite of his Jewish heritage."
	# parser_result =(parser.parse("The random person on the street eats meat.".split()))
	# for a in parser_result:
	# 	getNodes(a)
	# 	print("\u")
	# 	
	

	#	Read in the article and list of questions
	article_path = sys.argv[1]
	question_list = sys.argv[2]

	#	Tokenize all sentences
	sentence_pool = []

	article = open(article_path).read()
	paragraphs = [p for p in article.split('\n') if p]
	for paragraph in paragraphs[1:len(paragraphs)]: #	Skip the title
		sentences = sent_tokenize(paragraph)
		for sentence in sentences:
			sentence_tokenized = [a.lower() for a in word_tokenize(sentence)]
			sentence_pool.append(sentence_tokenized)
			
	#	Answer questions in the quesiton list
	count = 0

	#	Read in the lemmatized the sentences
	sentences_pool_lemmatized = []

	# Uncomment if the lemmatized sentence pool hasn't been generated yet
	# This step takes a long time, so you only need to run lemmatizaiton onece and you can load
	# the lemmatized setences from file to try different things after 
	# with open('sentences_pool_lemmatized.csv','w') as f:
	# 	writer = csv.writer(f,delimiter="\t")
	# 	for sent in sentence_pool:
	# 		sent = [a for a in sent if a != '\t']
	# 		sentences_lemmatized = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in pos_tag(sent)]
	# 		sentences_pool_lemmatized.append(sentences_lemmatized)
	# 		writer.writerow(sentences_lemmatized)
	
	# with open('sentences_pool_lemmatized.csv') as f:
	# 	for line in f:
	# 		line = [a.lower() for a in line.strip().split("\t")]
	# 		sentences_pool_lemmatized.append(line)

	with open(question_list) as f:
		#	For each question on the list
		for question in f:
			count += 1
			question_tokenized = word_tokenize(question)
			question_tokenized_lower = [a.lower() for a in question_tokenized]
			question_start = question_tokenized_lower[0]


			if question_start in ['when','where','why','which','who','how','do','does','did']:
				continue	
			
			
			#	Seperate question words and question content
			# filtered_list = [a for a in string.punctuation]
			filtered_list = ['?','when', 'what','where','what','why','which','who','how','do','does','did','a','the','an']
			question_content = [a for a in question_tokenized_lower if a not in filtered_list]

			#	Lemmatize the question
			#question_lemmatized = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in pos_tag(question_content)]
			
			
			#	Find the most similar sentences in the pool 
			max_similarity = None
			most_similar_sent = [] #	We need to consider ties

			for sent_idx in range(len(sentence_pool)):
				sent = sentence_pool[sent_idx]
				
				#similarity_score = jaccard_similarity(sent,question_content)+similarity(sent,question_content)
				if question_start == 'why':
					similarity_score = similarity_why(sent,question_content)
				else:
					similarity_score = similarity(sent,question_content, 0.8)
				
				if max_similarity == None:
					max_similarity = similarity_score
					#	Append the origin un-lemmatized sentence
					most_similar_sent.append(sentence_pool[sent_idx])
				elif similarity_score > max_similarity:
					max_similarity = similarity_score
					most_similar_sent.append(sentence_pool[sent_idx])
				else:
					pass
			

			# print((most_similar_sent))
			# print

			#	Now, build the answer from the retrieved sentence
			same_word = set(most_similar_sent[0])
			for s in most_similar_sent[1:]:
				same_word.intersection_update(s)
			
			#	Find the most relevant sentence
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
			print(question)
			print(max_similar_sent)
			#	Build answer based on different type of question
			answer = "NULL"

			#	Yes/No question: answer should contain only yes or no.	
			
			if question_start in ["is","was","are","were","do","does","did","have","has","had"]:
				#	First, convert sentence into a declarative sentence
				if max_similarity_2 == 0:
					answer = "No"
				else:
					question_parse = parser.parse(question_tokenized)
					print "--Q:", question_tokenized
					# max_similar_parse = parser.parse(max_similar_sent)
					for parse in question_parse:
						# print(parse)
						verb = parse[0][0].leaves()
						sub = (parse[0][1].leaves())
						obj = (parse[0][2].leaves())
						#substring = " ".join((sub+verb+obj))
						# If yes, most of the words in objects should be in the original sentence
						verb = [a.lower() for a in verb]
						obj = [a.lower() for a in obj]

						print verb, sub, obj
						print "--A:", max_similar_sent
						max_similar_parse = parser.parse(max_similar_sent)
						for mparse in max_similar_parse:
							# print(parse)
							try: 
								averb = mparse[0][1][0].leaves()
								averb = [a.lower() for a in averb]
								print averb
							except:
								answer = "no"
								break
							# asub = (mparse[0][1].leaves())
							aobj = (mparse[0][2].leaves())
							aobj = [a.lower() for a in aobj]
							# averb = [a.lower() for a in averb]
							# print averb, asub, aobj

							if verb == averb and aobj[0] != 'not':
								answer = "yes"
							else:
								# Handle: is - isn't and is - is not and not relavant
								answer = "no"
					# question_parse = parser.parse(question_tokenized)
					# for parse in question_parse:
					# 	# print(parse)
					# 	verb = parse[0][0].leaves()
					# 	sub = (parse[0][1].leaves())
					# 	obj = (parse[0][2].leaves())
					# 	#substring = " ".join((sub+verb+obj))
					# 	# If yes, most of the words in objects should be in the original sentence
					# 	obj = [a.lower() for a in obj]
					# 	if float(len(intersection(obj,max_similar_sent))) / len(obj)  >= 0.8:
					# 		answer = "Yes"
					# 	else:
					# 		answer = "No"
						
					#	TODO: parse candidate sentence
					# answer = "No"
					# similar_sent_parse = parser.parse(max_similar_sent)
					# for parse in similar_sent_parse:
					# 	verb_ = parse[0][0].leaves()
					# 	sub_ = (parse[0][2].leaves())
					# 	obj_ = (parse[0][1].leaves())
					
			elif question_start == 'why':
				max_similar_sent_str = " ".join(max_similar_sent)
				reason_idx = max_similar_sent_str.index('because of')
				answer = max_similar_sent_str[len('because of'):len(max_similar_sent_str)]
				if reason_idx == -1:
					reason_idx = max_similar_sent_str.index('because')
					answer = max_similar_sent_str[len('because'):len(max_similar_sent_str)]
				if reason_idx == -1:
					reason_idx = max_similar_sent_str.index('for')
					answer = max_similar_sent_str[len('for'):len(max_similar_sent_str)]
				if reason_idx == -1:
					answer = "NULL"


			elif question_start == 'when':
				found_DATE = False
				max_similar_sent_tag = ner.tag(max_similar_sent)
				for pair in max_similar_sent_tag:
					if pair[1] == 'DATE' or pair[1] == 'TIME':
						answer = pair[0]
						found_DATE = True
				if not found_DATE:
					#TODO: deal with this situation
					timeFlag = ["year", "month", "day", "hour", "minute", 'second'] #And more
					if "during" in max_similar_sent:
						duringIndex = max_similar_sent.index("during")
						answer = []
						for i in max_similar_sent[duringIndex:]:
							answer.append(i)
							if lemmatizer.lemmatize(i) in timeFlag:
								answer.append(i)
								break
						if i == len(max_similar_sent[duringIndex:]):
							answer = []
					#TODO: other situations
			
			
			elif question_start == 'who':
				max_similar_sent_tag = ner.tag(max_similar_sent)
				found_PERSON = False
				for pair in max_similar_sent_tag:
					if pair[1] == 'PERSON':
						answer = pair[0]
						found_PERSON = True
				if not found_PERSON:
					#TODO: deal with this situation
					max_similar_parse = parser.parse(max_similar_sent)
					for mparse in max_similar_parse:
						# print(parse)
						asub = (mparse[0][1].leaves())
						answer = [a.lower() for a in asub]
						#TODO: other situations


			elif question_start == 'where':
				found_LOCATION = False
				max_similar_sent_tag = ner.tag(max_similar_sent)
				for pair in max_similar_sent_tag:
					if pair[1] == 'LOCATION' or pair[1] == 'LOCATION':
						answer = pair[0]
						found_LOCATION = True
				if not found_LOCATION:
					#TODO: deal with this situation
					pass


				
			#For what, which, and how, and others
			else:
				#print(count,question)
				try:
					question_parse = parser.parse(question_tokenized)
					for parse in question_parse:
						#print(parse)
						verb = parse[0][1].leaves()
						sub = (parse[0][1][1].leaves())
						#obj = (parse[0][2].leaves())
						#print(verb,sub)

					similar_sent_parse = parser.parse(max_similar_sent)
					for parse in similar_sent_parse:
						# print(parse)
						answer = parse[0][1][1].leaves()				
				except:
					pass
					#TODO: deal with this situation


			#Capitalize first letter
			answer = " ".join(answer).title()

			print(answer)
			# > answer.txt

			
		

if __name__ == '__main__':
	main()
