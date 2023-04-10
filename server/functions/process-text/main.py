from flask import escape, jsonify
import functions_framework
from gensim import corpora, models
from nltk.tokenize import sent_tokenize
import nltk
import pyLDAvis.gensim_models

nltk.download('punkt')

@functions_framework.http
def process_text(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'text' in request_json:
        text = request_json['text']
    else:
        text = ''

    # # Read in the text from a file
    # with open("book_of_mormon_full_text.txt", "r") as f:
    #     text = f.read()

    # Split the text into sentences
    sentences = sent_tokenize(text)

    # Preprocess the text
    stop_words = [
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as',
        'at', 'be', 'became', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'came',
        'can',
        'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
        'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he',
        'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how',
        'how\'s',
        'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself',
        'let\'s',
        'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
        'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d',
        'she\'ll',
        'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'suppose', 'than', 'that', 'that\'s', 'the', 'thee',
        'thy',
        'thine', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d',
        'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'thou', 'through', 'to', 'too', 'under', 'until', 'unto',
        'up',
        'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s',
        'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with',
        'won\'t', 'would', 'wouldn\'t', 'you', 'you?', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours',
        'yourself', 'yourselves'
    ]
    texts = []
    for sentence in sentences:
        tokens = sentence.lower().split()
        tokens = [token for token in tokens if token not in stop_words]
        texts.append(tokens)
    dictionary = corpora.Dictionary(texts)

    # Create a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # Train the LDA model
    print('Training the LDA model ...')
    num_topics = 10
    lda_model = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=40)

    # Print the top words in each topic
    topics = []
    for i, topic in lda_model.show_topics(num_topics=num_topics):
        topic_dict = {}
        topic_dict["index"] = i
        topic_dict["words"] = topic
        topics.append(topic_dict)

    # Visualize the topics
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary=lda_model.id2word)

    response = jsonify({
        "topics": topics,
        "visualization": vis.to_dict(),
    })
    return response
