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
	return answer
					
