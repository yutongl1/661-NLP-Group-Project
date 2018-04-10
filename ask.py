#!/usr/bin/env python


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
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
import string
from collections import Counter
import random 
from ginger_python2 import get_ginger_result 
import re




# stanford_pos = 'stanford/stanford-postagger-full-2015-04-20/'
# stanford_pos_model = stanford_pos + 'models/english-left3words-distsim.tagger'
# stanford_pos_jar = stanford_pos + 'stanford-postagger.jar'
# st_pos = StanfordPOSTagger(model_filename=stanford_pos_model, path_to_jar=stanford_pos_jar)

stanford_parser = 'stanford/stanford-parser-full-2015-04-20/'
eng_model_path = stanford_parser + "englishPCFG.caseless.ser.gz"
stanford_parser_model = stanford_parser + 'stanford-parser-3.5.2-models.jar'
stanford_parser_jar = stanford_parser + 'stanford-parser.jar'
st_parser = StanfordParser(model_path = eng_model_path, path_to_models_jar=stanford_parser_model, path_to_jar=stanford_parser_jar)


stanford_ner = 'stanford/stanford-ner-2015-04-20/'
stanford_ner_model1 = stanford_ner + 'classifiers/english.all.3class.distsim.crf.ser.gz'
stanford_ner_model2 = stanford_ner + 'classifiers/english.muc.7class.distsim.crf.ser.gz'
stanford_ner_jar = stanford_ner + 'stanford-ner.jar'
st_ner1 = StanfordNERTagger(model_filename=stanford_ner_model1, path_to_jar=stanford_ner_jar)
st_ner2 = StanfordNERTagger(model_filename=stanford_ner_model2, path_to_jar=stanford_ner_jar)

punctuation = ['\\','/', ';','@', '?', '^','~', '`', '|']
lmtzr = WordNetLemmatizer()


# -------- Yutong's Editing ---------
# 1. Post processing: 
	# 1) capitalize/decapitalize 
	# 2) Symbols 
	# 3) Remove comma phrases and paranthese phrases
# 2. Expand question: replace adjectives with synonyms and antonyms (yes/no questions only)
	# 1) If the adjective word is a number or ordinal, try to find the numerical form in the WordNet and replace it 
	# 	(e.g. fourth -> 4th, nine -> 9. "first" and "one" are excluded)
	# 2) Otherweise Use WordNet to choose the most common synonyms and antonyms of the word 
# 3. Filtering 
	# 1) Automatically check grammar by calling Ginger API
	# 2) Filter out questions too long or too short (TO DO: Add Priority; Try Except)
	# 3) Sample (Random Sample or Select Shorter?)
	# 4) TO DO: Manual Evaluation



# ------------------------------------------
# Problem to solve
# 1. NER Tag is not really good at recognizing PERSON. For example, Dempsey in set1/a1.txt is replaced by "what" all the time
#	 Possible solution: a list of common names from U.S. Census data.
# 4. Why question
# 5. When replacing synonyms and antonyms, have not yet considered the case "XX and YY". 
# 6. Pronouns in sentences (e.g.  What scored his first and the eventual match-winning goal for Tottenham in a 3-2 win over Manchester United?) 
#    Possible solution: co-reference resolution


# -----------------------------------------

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
	pronoun = ["There", "It", "I", "You", "She", "He","They", "Its", "His", "Her", "Their", "Your", "My","These", "Those", "This", "That"]
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

	generated_question = []

	
	for S in parser_result: 
		# Work on wh-subject sentences. ["Who", "Which Organization", "Yes No", "Where"]
		if S[0][0].label() == 'NP' and S[0][1].label() == 'VP':
			subject_words = S[0][0].leaves()
			ner_tags =  map(lambda x: x[1], st_ner1.tag(subject_words))
			NP_ner_tag = Counter(ner_tags) 
			if len(NP_ner_tag) > 2:
				# TO DO: Majority Vote
				# print "=============== Tell Me =========="
				# print sentence
				pass

			if "PERSON" in NP_ner_tag:
				# print "============= Person =============="
				subject_question = "Who " + ' '.join(S[0][1].leaves()) + "?"
			elif "ORGANIZATION" in NP_ner_tag:
				# TO DO: Need to figure out plural or singular
				# print "============== organization ============="
				subject_question = "Which organization " + ' '.join(S[0][1].leaves()) + "?"
				# print "Which organization " + ' '.join(S[0][1].leaves()) + "?"
			elif "O" in NP_ner_tag and len(NP_ner_tag) == 1:
				# print "============== what? ============="
				subject_question = "What " + ' '.join(S[0][1].leaves()) + "?"
				# print question


			generated_question.append(subject_question)
			# Replace adjectives in the oringinal question with synonyms. Add to candidate
			replace_synonyms = expand_synonyms(subject_question)
			if replace_synonyms:
				generated_question.append(replace_synonyms)

				
				
			# Yes No Question
			be = ['is', 'are', 'was', 'were']
			VB = S[0][1][0]
			if len(VB.leaves()) == 1:
				# ======== Yutong's Modification ==========
				# If subject contains a person, then do nothing (very heuristic)
				# otherwise, we need to decapitalize the first letter of the subject 
				# When making a yes-no question
				subject = ' '.join(S[0][0].leaves()).replace("The", "the")
				if "PERSON" not in NP_ner_tag and "LOCATION" not in NP_ner_tag and "ORGANIZATION" not in NP_ner_tag: 
					subject = subject.lower()
				# ======== Yutong's Modification ==========

				if VB.leaves()[0] in be:
					# Capitalize the BE word + The first NP Phrase (Subject) + Everything after the Be word. 
					yesno_question = VB.leaves()[0].capitalize() + ' ' + subject + ' ' +  ' '.join(S[0][1][1].leaves()) + '?'
					# print question
				else:
					# Do 
					if VB.label() == "VBP":
						yesno_question = "Do " + subject + ' ' +  lmtzr.lemmatize(VB.leaves()[0],'v') + ' ' + ' '.join(S[0][1][1].leaves()) + '?'						
					# Does
					elif VB.label() == "VBZ":
						yesno_question = "Does " + subject + ' ' + lmtzr.lemmatize(VB.leaves()[0],'v') + ' ' + ' '.join(S[0][1][1].leaves()) + '?'
					# Did
					elif VB.label() == "VBD":
						yesno_question = "Did " + subject + ' ' + lmtzr.lemmatize(VB.leaves()[0],'v') + ' ' + ' '.join(S[0][1][1].leaves()) + '?'
					# print question

				generated_question.append(yesno_question)

				replace_synonyms = expand_synonyms(yesno_question)
				if replace_synonyms:
					generated_question.append(replace_synonyms)


				# For yes-no questions, can replace adjectives with antonyms
				replace_antonyms = expand_antonyms(yesno_question)
				if replace_antonyms:
					generated_question.append(replace_antonyms)				



            # Where and When
			words = S[0].leaves()
			tags =  map(lambda x: x[1], st_ner2.tag(words))
			tagDict = Counter(tags) 
			# Where
			if "LOCATION" in tagDict:
				if len(VB.leaves()) == 1:
					q_loc = ["Where"]
					if VB.leaves()[0] in be:
						q_loc.append(VB.leaves()[0])
						for i in range(len(words) - 1):
							if words[i] == VB.leaves()[0] or tags[i] == "LOCATION" or tags[i + 1] == "LOCATION":
								pass
							else:
								if i == 0 and (not tags[i] == "PERSON") and (not tags[i] == "ORGANIZATION"):
									q_loc.append(words[i].lower())
								else:
									q_loc.append(words[i])
					else:
						# Do 
						if VB.label() == "VBP":
							q_loc.append("do")
						# Does
						if VB.label() == "VBZ":
							q_loc.append("does")
						# Did
						if VB.label() == "VBD":
							q_loc.append("did")

						for i in range(len(words) - 1):
							if tags[i] == "LOCATION" or tags[i + 1] == "LOCATION":
								pass
							elif words[i] == VB.leaves()[0]:
								q_loc.append(lmtzr.lemmatize(VB.leaves()[0],'v'))
							else:
								if i == 0 and (not tags[i] == "PERSON") and (not tags[i] == "ORGANIZATION"):
									q_loc.append(words[i].lower())
								else:
									q_loc.append(words[i])
					# print "============= Location =============="
					pp_question = ' '.join(q_loc) + "?"
					# print question
					generated_question.append(pp_question)

					replace_synonyms = expand_synonyms(pp_question)
					if replace_synonyms:
						generated_question.append(replace_synonyms)

			# When
			if "DATE" in tagDict or "TIME" in tagDict:
				if len(VB.leaves()) == 1:
					q_loc = ["When"]
					if VB.leaves()[0] in be:
						q_loc.append(VB.leaves()[0])
						for i in range(len(words) - 1):
							if words[i] == VB.leaves()[0] or tags[i] == "DATE" or tags[i] == "TIME" or tags[i + 1] == "DATE" or tags[i + 1] == "TIME":
								pass
							else:
								if i == 0 and (not tags[i] == "PERSON") and (not tags[i] == "ORGANIZATION"):
									q_loc.append(words[i].lower())
								else:
									q_loc.append(words[i])
					else:
						# Do 
						if VB.label() == "VBP":
							q_loc.append("do")
						# Does
						if VB.label() == "VBZ":
							q_loc.append("does")
						# Did
						if VB.label() == "VBD":
							q_loc.append("did")

						for i in range(len(words) - 1):
							if tags[i] == "DATE" or tags[i] == "TIME" or tags[i + 1] == "DATE" or tags[i + 1] == "TIME":
								pass
							elif words[i] == VB.leaves()[0]:
								q_loc.append(lmtzr.lemmatize(VB.leaves()[0],'v'))
							else:
								if i == 0 and (not tags[i] == "PERSON") and (not tags[i] == "ORGANIZATION"):
									q_loc.append(words[i].lower())
								else:
									q_loc.append(words[i])
					
					# print "============= Time =============="
					pp_question = ' '.join(q_loc) + "?"
					# print question
					generated_question.append(pp_question)
					replace_synonyms = expand_synonyms(pp_question)
					if replace_synonyms:
						generated_question.append(replace_synonyms)
	return generated_question


def removePhrases(sent):

	# Remove things between paranthesis
	sent = re.sub("[\(\[].*?[\)\]]", "", sent)

	
	sent_word = word_tokenize(sent)
	idx_comma = []


	for i in xrange(len(sent_word)):
		if sent_word[i] == ",":
			if (i > 0 and any(c.isdigit() for c in sent_word[i-1])) or (i < len(sent_word) - 1) and any(c.isdigit() for c in sent_word[i+1]):
				continue
			else:
				idx_comma.append(i)
	if not len(idx_comma):
		return sent
	elif len(idx_comma) == 1:
		sent = ' '.join(sent_word[:idx_comma[0]]) + "?"
	elif len(idx_comma) == 2:
		sent = ' '.join(sent_word[:idx_comma[0]] + sent_word[idx_comma[1] + 1:]) 

	return sent
		


# Remove redundant phrases to make the questions consice
def post_processing(question):
	# print "Original:", question
	symbols = {" 's" : "'s", " ," : ",", "`` ":  "''", " ''": "''", " :" : ":", "  ": " ", "$ ": "$" }

	question = question.replace("-lrb- ", "(").replace(" -rrb-", ")").replace("-LRB- ", "(").replace(" -RRB-",")")

	question = removePhrases(question)

	
	for key in symbols:		
		if key in question:
			question = question.replace(key, symbols[key])				
	# print "Processed:", question
	return question

def expand_synonyms(sentence):
	text = word_tokenize(sentence)
	word_tag = pos_tag(text)
	replace_number = False
	#word_tag = st_pos.tag(sentence.split())              

	for (i, (word, tag)) in enumerate(word_tag):
		if tag == "JJ":
			jj_idx = i
			synonyms = []
			for syn in wn.synsets(word):
				for lemma in syn.lemmas():
					synonyms.append(lemma.name().replace("_"," "))

			synonyms = filter(lambda a: a != word, synonyms)

			if synonyms: 
				# print "==== Replacing Synonyms ========"
				# print "Original:", sentence
				for s in synonyms:
					if any(char.isdigit() for char in s) and word != "one" and word != "first":
						replace_word = s
						replace_number = True
						break
				if not replace_number:
					replace_word = Counter(synonyms).most_common()[0][0]
				sentence = sentence.replace(word, replace_word)
				# print "Replaced:", sentence
				# print "==== End Replacing Synonyms ========"
				return sentence
	return None


def expand_antonyms(sentence):

	text = word_tokenize(sentence)
	word_tag = pos_tag(text)
	#word_tag = st_pos.tag(sentence.split())

	for (i, (word, tag)) in enumerate(word_tag):
		if tag == "JJ":

			jj_idx = i
			antonyms =  []
			for syn in wn.synsets(word):
				for lemma in syn.lemmas():
					if lemma.antonyms():
						antonyms.append(lemma.antonyms()[0].name())
			if antonyms:
				# print "==== Replacing antonyms ========"
				# print "Original:", sentence
				sentence = sentence.replace(word, Counter(antonyms).most_common()[0][0])
				# print "Replaced:", sentence
				# print "==== End Replacing antonyms ========"
				return sentence
	return None


def filterQuestions(question_list):
	# print "============== Filtering ============"
	i = 0
	questions_priority1 = []
	questions_priority2 = []
	# Filter our ungrammatical questions in the candidate set
	# May filter our false negatives, but at least most of them are correct. 
	for q in question_list:
		try:
		#if True:
			# i += 1
			q = post_processing(q)
			if get_ginger_result(q) and len(q.split()) > 4 and len(q.split()) < 25:
				# i += 1
				# print "Good Question %d" %i, q
				questions_priority1.append(q)
			else:
				# i += 1	
				# print "Bad Question %d" %i, q
				questions_priority2.append(q)
				continue
		except:
			continue
	return questions_priority1, questions_priority2


def main():
	sentence_word = sentenceCandidate(sys.argv[1])
	nquestion = int(sys.argv[2])
	generated_questions = []
	
	# Loop through all tokenized sentence, generate questions if possible 
	# Add to candidate set generated_questions
	for sent in sentence_word:
		try:
			questions_from_sent = parse_sentence(sent)
			if questions_from_sent != []:
				generated_questions += questions_from_sent
				if len(generated_questions) > 100:
					break
		except Exception as e:
			print "ERROR in parsing:", e
			pass

	try:

		questions_priority1, questions_priority2 = filterQuestions(generated_questions)
		questions_priority1 = set(questions_priority1)
		questions_priority2 = set(questions_priority2)

	except Exception as e2:
		print "Error in Filtering:", e
		pass
	

	print " ======== Generated %d good questions and %d not so good questions =======" % (len(questions_priority1), len(questions_priority1))
	print "========= Sampling %d questions" % nquestion
	
	# If ginger filter out too many questions, we sample some from the excluded ones. 
	if len(questions_priority1) < nquestion:
		sampled_question = list(questions_priority1) + random.sample(questions_priority2, nquestion - len(questions_priority1))
	else:
		sampled_question = random.sample(questions_priority1, nquestion)

	for (i, q) in enumerate(list(sampled_question)):
		print "Question %d" %i, q



# Grammar Check use Ginger API: https://github.com/zoncoen/python-ginger
	 	
			 
main()
