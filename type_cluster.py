import json
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
import nltk
from nltk.parse.stanford import StanfordDependencyParser
import numpy as np
from gensim.models import word2vec
from gensim.models import KeyedVectors
import re

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# 1. 读取context文件


def load_data(path):
    with open(os.path.join(path, 'entity2wikidata.json'), 'r') as f:
        json_dict = json.load(f)

    id2description = {}
    descriptions = []  # list for computation
    for k, v in json_dict.items():
        # 可能存在none的情况，需要特殊处理，占比也不多
        des = v['description'] if v['description'] else ''
        id2description[k] = des
        descriptions.append(des)

    return id2description, descriptions

# 2. 获得embedding


def mean_podding(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# sentences = ['This is an example sentence', 'Each sentence is converted']


def load_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.cuda()

    return tokenizer, model


def get_embedding(model, tokenizer, sentences):

    encoded_input = tokenizer(sentences, padding=True,
                              truncation=True, return_tensors='pt').to('cuda')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentences_embeddings = mean_podding(
        model_output, encoded_input['attention_mask'])

    return sentences_embeddings

# 3. 聚类，考虑根据纯text信息或者用个bert啥的


def tsne_visualize(sentence_embeddings, freq=None):
    reducted_embeddings = TSNE(
        n_components=3, learning_rate='auto', init='random', n_iter=10000).fit_transform(sentence_embeddings)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if freq is None:
        ax.scatter(reducted_embeddings[:, 0], reducted_embeddings[:, 1])
    else:
        freq = np.sqrt(freq)
        freq = freq / np.max(freq)
        ax.scatter(reducted_embeddings[:, 0], reducted_embeddings[:, 1], reducted_embeddings[:, 2], alpha=freq)
        
    plt.savefig('vis.png')
    return

def is_noun(pos): return pos == 'NN' or pos == 'NNS'

if __name__ == "__main__":

    path = 'data/FB15K/'
    id2description, descriptions = load_data(path)

    words = []
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    for des in descriptions:
        # new_words = des.split()
        # new_words = [word.strip(",.'") for word in new_words]
        des = des.replace('-', ' ')
        des = des.replace('–', ' ') # data is not clean
        des = des.replace('/', ' ')
        tokenized = nltk.word_tokenize(des)
        # modified = []
        # if have both, like singer, song writer, we only remain the last one, considering the case like co-founder
        # for word in tokenized:
        #     word = word.strip('–/') # contains 1999-, and/or
        #     word = re.split('–/', word)
        #     if len(word) > 1:
        #         print(word)
        #         if re.match(r'\D*\d+', word):
        #             continue # filter out all numbers
        #     modified.append(word[-1])
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
        nouns = [lemmatizer.lemmatize(word, 'n') for word in nouns]

        words.extend(nouns)

    counter = Counter(words)
    word2count = OrderedDict(counter.most_common())
    words = list(word2count.keys())

    # model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    # tokenizer, model = load_model(model_name)

    FASTTEXTFILE="~/word2vec/wiki-news-300d-1M.vec"
    model=KeyedVectors.load_word2vec_format(FASTTEXTFILE,limit=500000)
    word_embeddings = []
    freqs = []
    for word in words:
        if word in model.key_to_index.keys():
            word_embeddings.append(model[word]) 
            freqs.append(word2count[word])
        else:
            print(word)


    # batch_size = 1000
    # batch_begin = 0

    # des_embeddings = []

    # while batch_begin < len(descriptions):
    #     batch_data = descriptions[batch_begin:batch_begin + batch_size]
    #     des_embeddings.append(get_embedding(model, tokenizer, batch_data))
    #     batch_begin += batch_size

    # des_embeddings = torch.cat(des_embeddings, dim=0)

    # word_embeddings = []

    # while batch_begin < len(words):
    #     batch_data = words[batch_begin:batch_begin + batch_size]
    #     word_embeddings.append(model(batch_data))

    #     # word_embeddings.append(get_embedding(model, tokenizer, batch_data))
    #     batch_begin += batch_size

    # word_embeddings = torch.cat(word_embeddings, dim=0)
    word_embeddings = np.stack(word_embeddings, axis=0)
    tsne_visualize(word_embeddings, freqs)

    print('over')
