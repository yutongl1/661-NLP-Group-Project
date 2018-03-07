import nltk

def main(): 

	files = ['a' + str(i) + '.txt' for i in xrange(1,11)]

	for file in files:
		noun_count = 0 
		with open('set3/'+file) as f:
			for line in f:
				line = nltk.word_tokenize(line.decode('utf-8'))
				word_tag = nltk.pos_tag(line)
				for token in word_tag:
					tag = token[1]
					if tag.startswith("N"):#See NLTK pos tagger documentation
						noun_count += 1
		#	Output the noun count for current file
		print("Number of nouns in "+file+" "+str(noun_count))



if __name__ == "__main__":
    main()


'''
a1.txt 5061
a2.txt 2714
a3.txt 1033
a4.txt 2764
a5.txt 2435
a6.txt 3194
a7.txt 2333
a8.txt 2563
a9.txt 2219
a10.txt 3201
'''


