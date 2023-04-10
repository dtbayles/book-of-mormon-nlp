from flask import jsonify, request, Flask, Response
from gensim import corpora, models
from nltk.tokenize import sent_tokenize
import nltk
import pyLDAvis.gensim_models
import json

nltk.download('punkt')

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def process_text():
    req_data = request.get_json()
    text = req_data['text']

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
        topics.append({
            "id": i,
            "top_words": topic
        })

    # Visualize the topics
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary=lda_model.id2word)

    response_data = {
        "topics": topics,
        "visualization": json.loads(pyLDAvis.prepared_data_to_json(vis))
    }
    response = Response(response=json.dumps(response_data), status=200, mimetype="application/json")
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
    app.run()
