def ansWhen(max_similar_sent):
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

