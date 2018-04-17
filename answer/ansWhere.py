def ansWhere(max_similar_sent):
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