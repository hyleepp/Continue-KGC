import json
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
import nltk
# from nltk.parse.stanford import StanfordDependencyParser
import numpy as np
from gensim.models import word2vec
from gensim.models import KeyedVectors
import brewer2mpl
import re
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import pickle


# download the required resources
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('dependency_treebank')

# load the dependency parser
dep_parser = nltk.parse.DependencyGraph

# create a sentence to parse
sentence = "The quick brown fox jumps over the lazy dog."

# tokenize the sentence
tokens = nltk.word_tokenize(sentence)

# tag the tokens
pos_tags = nltk.pos_tag(tokens)

# # parse the sentence
# dep_graph = dep_parser.parse(pos_tags)

# # print the parsed dependencies
# print(dep_graph.to_conll(10))


# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')


def load_data(path):
    # key means sth like /m/012qdp
    # qid means id in wiki, like Q5
    # idx means id in triples, int number, i-th entity
    with open(os.path.join(path, 'entity2wikidata.json'), 'r') as f:
        entity2wiki = json.load(f)
        key2qid = {}
        for key, value in entity2wiki.items():
            key2qid[key] = value['wikidata_id']


    with open(os.path.join(path, 'entity2id.txt'), 'r') as f:
        lines = f.readlines()
        key2idx = {} 
        for line in lines:
            key, idx = line.strip().split()
            idx = int(idx)
            key2idx[key] = idx
        
    idx2class = {}
    class_names = []

    with open(os.path.join(path, "entity_type.json"), 'r') as f:
        qid2class = json.load(f)
    for k, v in entity2wiki.items():
        des = v['description'] if v['description'] else ''
        wiki_id = v['wikidata_id']
        class_name = qid2class[wiki_id]['parent_entity_name']
        idx = key2idx[k]
        idx2class[idx] = class_name
        class_names.append(class_name)
    
    return idx2class, class_names 

# 2. attaining embedding


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

# 3. clustering, considering textual information 


def tsne_visualize(sentence_embeddings, freq=None, labels=None):
    # distance_matrix = pairwise_distances(sentence_embeddings, sentence_embeddings, metric='cosine', n_jobs=-1)
    palette = 'hsv'  
    cmap = plt.get_cmap(palette)
    reducted_embeddings = TSNE(
        n_components=2, learning_rate='auto', init='random', n_iter=1000, metric='cosine').fit_transform(sentence_embeddings)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()

    if freq is None:
        ax.scatter(reducted_embeddings[:, 0], reducted_embeddings[:, 1], c=labels, cmap=cmap)
        # ax.scatter(reducted_embeddings[:, 0], reducted_embeddings[:, 1], reducted_embeddings[:, 2], c=labels)
    else:
        freq = np.sqrt(freq)
        freq = freq / np.max(freq)
        ax.scatter(reducted_embeddings[:, 0], reducted_embeddings[:, 1], alpha=freq, c=labels)
        
    plt.savefig('vis.png')
    return

def is_noun(pos): return pos == 'NN' or pos == 'NNS'

def generate_relation_filter(triples, id2class, n_rel):
    # generate the relation filter based on entity class (like human).

    class_filte_relation = defaultdict(set)
    for h, r, t in triples:
        try:
            # there may have some entity miss the corresponding descriptions
            h_class, t_class = id2class[h], id2class[t]
            class_filte_relation[h_class].add(r)
            class_filte_relation[t_class].add(r + n_rel)
        except:
            continue
    
    return class_filte_relation

        

    
if __name__ == "__main__":

    path = 'data/FB15K/'
    idx2description, descriptions = load_data(path)

    # filters
    # try to only reserve the first noun
    # filted = []
    # for description in descriptions:
    #     if not description:
    #         filted.append('') # none
    #     else:
    #         words = description.split() 
    #         pos_tags = nltk.pos_tag(words)
    #         filted_sent=  [word for word, pos in pos_tags if pos == 'NN']
    #         filted.append(' '.join(filted_sent))

    

    # words = []
    # lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    # for des in descriptions:
    #     # new_words = des.split()
    #     # new_words = [word.strip(",.'") for word in new_words]
    #     des = des.replace('-', ' ')
    #     des = des.replace('–', ' ') # data is not clean
    #     des = des.replace('/', ' ')
    #     tokenized = nltk.word_tokenize(des)
    #     # modified = []
    #     # if have both, like singer, song writer, we only remain the last one, considering the case like co-founder
    #     # for word in tokenized:
    #     #     word = word.strip('–/') # contains 1999-, and/or
    #     #     word = re.split('–/', word)
    #     #     if len(word) > 1:
    #     #         print(word)
    #     #         if re.match(r'\D*\d+', word):
    #     #             continue # filter out all numbers
    #     #     modified.append(word[-1])
    #     nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    #     nouns = [lemmatizer.lemmatize(word, 'n') for word in nouns]

        # words.extend(nouns)

    # counter = Counter(words)

    # test cover rate
    with open('data/FB15K/active_learning/0.9/init_triples.txt', 'r') as f:
        init_triples = [list(map(lambda x: int(x), line.strip().split())) for line in f.readlines()]
        
    with open('data/FB15K/active_learning/0.9/unexplored_triples.txt', 'r') as f:
        unexplored_triples = [list(map(lambda x: int(x), line.strip().split())) for line in f.readlines()]

    init_filter = generate_relation_filter(init_triples, idx2description, 1345)
    all_filter = generate_relation_filter(init_triples + unexplored_triples, idx2description, 1345)
    
    # count how much init_filter can cover
    count = 0
    for triple in unexplored_triples:
        h, r, t = triple
        try:
            if r in init_filter[idx2description[h]] or r + 1345 in init_filter[idx2description[t]]:
                count += 1
        except:
            continue
            count += 1 
    print(count / len(unexplored_triples))
    counter = Counter(descriptions)
    des2count = OrderedDict(counter.most_common())
    descriptions = list(des2count.keys())

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer, model = load_model(model_name)

    # FASTTEXTFILE="~/word2vec/wiki-news-300d-1M.vec"
    # model=KeyedVectors.load_word2vec_format(FASTTEXTFILE,limit=500000)
    # word_embeddings = []
    # freqs = []
    # for word in words:
    #     if word in model.key_to_index.keys():
    #         word_embeddings.append(model[word]) 
    #         freqs.append(word2count[word])
    #     else:
    #         print(word)


    batch_size = 1000
    batch_begin = 0

    des_embeddings = []

    while batch_begin < len(descriptions):
        batch_data = descriptions[batch_begin:batch_begin + batch_size]
        des_embeddings.append(get_embedding(model, tokenizer, batch_data))
        batch_begin += batch_size

    des_embeddings = torch.cat(des_embeddings, dim=0)

    # word_embeddings = []

    # while batch_begin < len(words):
    #     batch_data = words[batch_begin:batch_begin + batch_size]
    #     word_embeddings.append(model(batch_data))

    #     # word_embeddings.append(get_embedding(model, tokenizer, batch_data))
    #     batch_begin += batch_size

    # word_embeddings = torch.cat(word_embeddings, dim=0)
    # word_embeddings = np.stack(word_embeddings, axis=0)
    des_embeddings = des_embeddings.detach().cpu().numpy()
    clustering = AgglomerativeClustering(n_clusters=50).fit(des_embeddings)
    labels = clustering.labels_
    
    clusters = defaultdict(list)
    for i in range(len(descriptions)):
        c = labels[i]
        clusters[c].append(descriptions[i])
    
    labels = None
    tsne_visualize(des_embeddings, None, labels)

    print('over')
