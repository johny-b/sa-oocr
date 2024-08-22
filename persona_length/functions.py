import nltk

def avg_sentence_length(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    lengths = [len(sentence.split()) for sentence in sentences]
    return round(sum(lengths) / len(sentences))