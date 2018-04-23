from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk import pos_tag

stop_words = set(stopwords.words('english'))

def jaccard_similarity(a,b):
  a = set(a)
  b = set(b)
  c = a.intersection(b)
  return float(len(c)) / (len(a) + len(b) - len(c))

# (A & B / B)
def percent_diff(listA, listB):
  return len(set(listA)&set(listB)) / float(len(set(listB)))

#Return true if it can be considered as a candidate answer
# def check_answer_subject_diff(answer, subject_list):
#   for subject in subject_list:
#     print(answer, subject.leaves(), percent_diff(answer, subject.leaves()), 'candidate')
#     if percent_diff(answer, subject.leaves()) >= 0.5:
#       return False
#   print("True")
#   return True

def check_answer_question_diff(answer, sq):
  answer = [a for a in answer if a not in stop_words]
  sq = [a for a in sq.leaves() if a not in stop_words]
  print(answer, sq, percent_diff(sq,answer), 'candidate')
  if percent_diff(sq, answer) >= 0.3:
    return False

  return True


def searchLabel(cur, label, record):
  answer = None
  if cur.label() == label:
    # record.append(cur.leaves())
    record.append(cur)
  for i in cur:
    # print "--",    (i), isinstance(i, (str, unicode)), i
    if not isinstance(i, (str, unicode)) and i.label() == label:
      # record.append(i.leaves())
      record.append(i)
    else:
      if len(i):
        if isinstance(i[0], (str, unicode)):
          continue
        else:
          for j in i:
            searchLabel(j, label, record)



def ansWho(parser,tagger, question_tokenized_lower, max_similar_sent, max_paragraph):
  #Break sentence into multiple sentences if there are punctuation (clauses, ",", ":" etc.)
  max_similar_sent_str = " ".join(max_similar_sent)
  max_similar_sent_sub = max_similar_sent_str.split(",")

  question_parse = parser.parse(question_tokenized_lower)
  similar_sent_parse = parser.parse(max_similar_sent)
  answer = max_similar_sent
 

  for parse_q in question_parse:
    print("question parse")
    print(parse_q)
    try:
      # See if there's PERSON in the max_sim_sentences
      max_similar_sent_upper = [a.title() for a in max_similar_sent]
      max_similar_sent_ner_tagged = tagger.tag(max_similar_sent_upper)
      for pair in max_similar_sent_ner_tagged:
        answer_list = []
        if pair[1] == 'PERSON' and pair[0].lower() not in question_tokenized_lower:
          answer_list.append(pair[0])
          return answer_list
      
      #There's no PERSON int he max_sim_sentences, set the answer to the original sentences
      answer = max_similar_sent_upper

      #See if coref resolution needs to be done
      answer_pos_tagged = pos_tag(answer)
      
      if 'PRP' in [a[1] for a in answer_pos_tagged] or 'PRP$' in [a[1] for a in answer_pos_tagged]:
        print "coref"
        for sent in max_paragraph:
          before_ner = tagger.tag([a.title() for a in sent]) 
          print("before: ",before_ner)
          for pair in before_ner:
            answer_list = []
            if pair[1] == 'PERSON' and pair[0].lower():
              answer_list.append(pair[0])
              answer = answer_list
    except Exception,e:
      print str(e)

  
  return answer