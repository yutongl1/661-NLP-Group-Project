def ansYesNo(question, max_similar_sent, max_similarity):
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