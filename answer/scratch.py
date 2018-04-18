from nltk import word_tokenize

a = 'Are Kangaroos Shy?'
b = 'Kangaroos are shy and retiring by nature, and in normal circumstances present no threat to humans.'
c = 'Kangaroos are endemic to the continent of Australia , while the smaller macropods are found in Australia and New Guinea .'

def similarity(a,b):
	return len(set(a).intersection(set(b)))

if __name__ == '__main__':
	a = word_tokenize(a)
	b = word_tokenize(b)
	c = word_tokenize(c)

	print(similarity(a,b))
	print(similarity(a,c))