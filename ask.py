import nltk
from nltk import word_tokenize, sent_tokenize
import os
import sys
sys.path.append("../")
#import config
from nltk.parse import stanford
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from nltk.tokenize.stanford import StanfordTokenizer
import string
from collections import Counter

data_path = 'data/set3'

#count = 0

stanford_parser = 'stanford/stanford-parser-full-2015-04-20/'
eng_model_path = stanford_parser + "englishPCFG.caseless.ser.gz"
stanford_parser_model = stanford_parser + 'stanford-parser-3.5.2-models.jar'
stanford_parser_jar = stanford_parser + 'stanford-parser.jar'
st_parser = StanfordParser(model_path = eng_model_path, path_to_models_jar=stanford_parser_model, path_to_jar=stanford_parser_jar)


stanford_ner = 'stanford/stanford-ner-2015-04-20/'
stanford_ner_model = stanford_ner + 'classifiers/english.all.3class.distsim.crf.ser.gz'
stanford_ner_jar = stanford_ner + 'stanford-ner.jar'
st_ner = StanfordNERTagger(model_filename=stanford_ner_model, path_to_jar=stanford_ner_jar)

punctuation = ['\\','/', ';','@', '?', '^','~', '`', '|']

# Return [[w01,...,wn1],[w02,...,wn2],...]
def sentenceCandidate(file):
	with open(file) as f:
		text = unicode(f.read().decode('utf-8'))
	# Use the sentence tokenizer provided by NLTK to tokenize the sentence

	# Problem: sent_tokenize does not consider split on new lines!
	# Solution: Split on new line, then tokenize
	paragraphs = [p for p in text.split('\n') if p]
	text_sent_tokenized = []
	for paragraph in paragraphs:
		text_sent_tokenized.extend(sent_tokenize(paragraph))
	# Preprocess each sentences, filter out irrelevant sentences, and tokenize sentences into words

	sentence_word = []
	for sent in text_sent_tokenized:
		
		doNotProcessing = False
		for c in punctuation:
			if c in sent:
				doNotProcessing = True
		if doNotProcessing:
			continue
		
		sent_word_tokenized = word_tokenize(sent)
		sentence_word.append(sent_word_tokenized)
		#print(sent_word_tokenized)
	sentence_word.sort(key = len)
	pronoun = ["It", "I", "You", "She", "He","They", "Its", "His", "Her", "Their", "Your", "My","These", "Those", "This", "That"]
	count = 0
	result = []

	for sent in sentence_word:
		if (len(sent) < 60 and len(sent) > 10 and 
			sent[0][0].isalpha() and 
			sent[0][0].isupper() and 
			sent[0] not in pronoun and 
			sent[-1] == '.'):
			
			
			result.append(sent)

	return result

def parse_sentence(sentence):
	parser_result = st_parser.parse(sentence)

	#print sentence

	for S in parser_result: 

		# Work on wh-subject sentences. ["Who", "Which Organization"]
		if S[0][0].label() == 'NP' and S[0][1].label() == 'VP':
			subject_words = S[0][0].leaves()
			
			ner_tags =  map(lambda x: x[1], st_ner.tag(subject_words))
			NP_ner_tag = Counter(ner_tags) 

			if len(NP_ner_tag) > 2:
				# TO DO: Majority Vote
				print "=============== Tell Me =========="
				print sentence
				print NP_ner_tag

			if "PERSON" in NP_ner_tag :
				print "============= Person =============="
				print "Who " + ' '.join(S[0][1].leaves()) + "?"
				#print S


			elif  "ORGANIZATION" in NP_ner_tag:
				# TO DO: Need to figure out plural or singular
				print "============== organization ============="
				print "Which organization " + ' '.join(S[0][1].leaves()) + "?"
				#print S

			elif "O" in NP_ner_tag and len(NP_ner_tag) == 1:
				print "============== what? ============="
				print "What " + ' '.join(S[0][1].leaves()) + "?"
				#print S				
				
			# Yes No Question


		# TO DO: Work on wh-non-subject (PP phrases) ["When", "Where"]



def main():

	sentence_word = sentenceCandidate(sys.argv[1])
	
	for sent in sentence_word:
		parseTree = parse_sentence(sent)
	
		
		#if parseTree != None:
			 


main()
