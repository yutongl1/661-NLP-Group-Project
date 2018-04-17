def ansWhat(parser, question_tokenized_lower, max_similar_sent):
  question_parse = parser.parse(question_tokenized_lower)
  for parse in question_parse:
    #print(parse)
    verb = parse[0][1].leaves()
    sub = (parse[0][1][1].leaves())
    #obj = (parse[0][2].leaves())
    #print(verb,sub)

  similar_sent_parse = parser.parse(max_similar_sent)
  for parse in similar_sent_parse:
    # print(parse)
    answer = parse[0][1][1].leaves()
  return " ".join(answer)