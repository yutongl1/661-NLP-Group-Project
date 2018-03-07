from nltk.parse import stanford
from nltk.tokenize.stanford import StanfordTokenizer

from nltk.parse import stanford
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordNeuralDependencyParser

from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
import nltk


'''
print "========= Checking POS ========="
stanford_pos = 'stanford/stanford-postagger-full-2015-04-20/'
stanford_pos_model = stanford_pos + 'models/english-left3words-distsim.tagger'
stanford_pos_jar = stanford_pos + 'stanford-postagger.jar'

st_pos = StanfordPOSTagger(model_filename=stanford_pos_model, path_to_jar=stanford_pos_jar)
print st_pos.tag('What is the airspeed of an unladen swallow ?'.split())

'''
print "========= Checking NER ========="
stanford_ner = 'stanford/stanford-ner-2015-04-20/'
stanford_ner_model = stanford_ner + 'classifiers/english.all.3class.distsim.crf.ser.gz'
stanford_ner_jar = stanford_ner + 'stanford-ner.jar'

st_ner = StanfordNERTagger(model_filename=stanford_ner_model, path_to_jar=stanford_ner_jar)
#print st_ner.tag('Rami Eid is studying at Stony Brook University in New York'.split())
print st_ner.tag("Gandalf deduces Sauron will attack Gondor 's capital Minas Tirith , riding there with Pippin?".split())




print "========= Checking PARSER ========="
stanford_parser = 'stanford/stanford-parser-full-2015-04-20/'
eng_model_path = stanford_parser + "englishPCFG.caseless.ser.gz"
stanford_parser_model = stanford_parser + 'stanford-parser-3.5.2-models.jar'
stanford_parser_jar = stanford_parser + 'stanford-parser.jar'
st_parser = StanfordParser(model_path = eng_model_path, path_to_models_jar=stanford_parser_model, path_to_jar=stanford_parser_jar)
parser_result = (st_parser.raw_parse('Rami Eid is studying at Stony Brook University in Los Angeles'))


for S in parser_result:
	if S[0][0].label() == 'NP' and S[0][1].label() == 'VP':
		subject_words = S[0][0].leaves()
		print subject_words
		print st_ner.tag(subject_words)

    	'''
        if type(node) is nltk.Tree:
            if node.label() == ROOT:
                print "======== Sentence ========="
                print "Sentence:", " ".join(node.leaves())
            else:
                print "Label:", node.label()
                print "Leaves:", node.leaves()

            getNodes(node)
        else:
            print "Word:", node
        '''
