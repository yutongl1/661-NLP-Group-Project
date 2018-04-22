from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords

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



def ansWhat(parser, question_tokenized_lower, max_similar_sent):
  #Break sentence into multiple sentences if there are punctuation (clauses, ",", ":" etc.)
  max_similar_sent_str = " ".join(max_similar_sent)
  max_similar_sent_sub = max_similar_sent_str.split(",")

  question_parse = parser.parse(question_tokenized_lower)
  similar_sent_parse = parser.parse(max_similar_sent)
  answer = max_similar_sent
  first_NP = False
  answer_candidate = True

  for parse_q in question_parse:
    #Handle the case where the selected sentence is of the structure WHNP SQ (VBZ NP)
    print("question parse")
    print(parse_q)
    try:
      if parse_q[0][0].label().startswith("WHNP") and parse_q[0][1].label().startswith("SQ"):
        #Find the subject being asked
        subject_list = []
        searchLabel(parse_q[0][1],"NP",subject_list)
        #subject_list = [a.leaves() for a in subject_list]

        for n in range(len(parse_q[0][1])):
          if parse_q[0][1][n].label().startswith("VB"):
            question_v = " ".join(parse_q[0][1][n].leaves())
            if question_v == "'s":
              question_v = 'is'
            elif question_v == "'re":
              question_v = 'are'
            question_v = wordnet_lemmatizer.lemmatize(question_v, pos='v')
            

            for parse_a in similar_sent_parse:

              print("answer parse")
              print(parse_a)

              for i in range(len(parse_a[0])):
                
                #Check if the answer is at the beginning of the sentence
                #Only do this check if the sentence is not too complicated (for instance, sentences without too many ",")
                
                if parse_a[0][i].label().startswith("NP") and subject_list != [] and first_NP == False and max_similar_sent.count(",") < 4:
                  first_NP = True
                  #answer_candidate = check_answer_subject_diff(parse_a[0][i].leaves(), subject_list)
                  answer_candidate = check_answer_question_diff(parse_a[0][i].leaves(), parse_q[0][1])
                  if answer_candidate:
                    #Check if the next node is a matching "VP"
                    try:
                      if parse_a[0][i+1].label() == "VP":
                        for o in range(len(parse_a[0][i+1])):
                          if parse_a[0][i+1][o].label().startswith("VB"):
                            answer_v = " ".join(parse_a[0][i+1][o].leaves())
                            answer_v = wordnet_lemmatizer.lemmatize(answer_v, pos='v')
                            if answer_v == question_v:
                              answer = parse_a[0][i].leaves()
                              print("In the beginning")
                              return answer
                    except Exception, e:
                      print(str(e))

                #The answer is not in the beginning of the sentence, check the rest of the sentence to find matching answer
                if (parse_a[0][i].label().startswith("VP")):
                  
                  
                  #Handle the case where the selected sentence is of the structure VP NP
                  for k in range(len(parse_a[0][i])):
                    if (parse_a[0][i][k].label().startswith("N")):
                      if "," in parse_a[0][i][k].leaves():
                        possible_np = []
                        answer_list = []
                        searchLabel(parse_a[0][i][k], 'NP',possible_np)
                        for answer in possible_np:
                          answer = answer.leaves()
                          #if not check_answer_subject_diff(answer, subject_list):
                          if not check_answer_question_diff(answer, parse_q[0][1]):
                            continue
                          else:
                            answer_list.extend(answer)
                            print("answer_ list: ",answer_list)
                            answer = answer_list

                      else:
                        answer = parse_a[0][i][k].leaves()

                      
                      print("Not in the beginning, returned from VP NP")
                      return answer
                    

                  #It could also be VP PP
                  for k in range(len(parse_a[0][i])):
                    if (parse_a[0][i][k].label().startswith("PP")):
                      answer = parse_a[0][i][k].leaves()
                      print("returned from VP PP")
                      return answer
                  

                  print("complicated case")
                  S = []
                  searchLabel(parse_a[0][i][k], "S", S)
                  if len(S)>=1:
                    answer_tree = S[0]
                    #Remove potential conjunction from the S
                    if len(S[0]) > 1:
                      answer_tree = S[0][0]

                    #Get the S from the S
                    s = []
                    searchLabel(answer_tree,"S",s)
                    if len(s) > 0:
                      answer = s[0].leaves()
                      return answer
                    
                    #Or, Get the NP from the S
                    np = []
                    searchLabel(answer_tree, "NP", np)
                    if len(np) > 0:
                      answer = np[0].leaves()
                      return answer
                 

                  #TODO: handle "respectively" type of questions
                  #
                  #


      
    except Exception,e:
      print str(e)

  
  return answer