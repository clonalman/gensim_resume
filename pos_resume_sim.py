#! /usr/bin/env python2.6.6
#coding=utf-8
import logging
from gensim import corpora, models, similarities

def similarity(datapath, querypath, storepath):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    class MyCorpus(object):
        def __iter__(self):
            for line in open(datapath, encoding='utf8'):
                yield line.split()

    Corp = MyCorpus()
    print('============字典==============');
    dictionary = corpora.Dictionary(Corp)
    print(dictionary.token2id)
    #dictionary.save('./data.dic')

    print('============语料库(词包)============');
    corpus = [dictionary.doc2bow(text) for text in Corp]
    for doc in corpus: 
        print(doc)

    print('============tfidf模型（加权）==========');
    tfidf = models.TfidfModel(corpus)
    print(tfidf.dfs)
    print(tfidf.idfs)
    #tfidf.save("./tfidf.mdl")
 
    print('============tfidf值==========');
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        print(doc)

    print('============新文件==========');
    q_file = open(querypath, 'r', encoding='utf8')
    query = q_file.readline()
    q_file.close()
    vec_bow = dictionary.doc2bow(query.split())

    vec_tfidf = tfidf[vec_bow]
    print('=========vec_tfidf=========')
    print(vec_tfidf)

    print('==========相似矩阵========')
    index = similarities.MatrixSimilarity(corpus_tfidf)
    index.save('./tfidf.idx')

    print('=========sims=========')
    sims = index[vec_tfidf]
    print(sims)

    similarity = list(enumerate(sims))
    sim_file = open(storepath, 'w', encoding='utf8')
    for i in similarity:
        sim_file.write(str(i)+'\n')
    sim_file.close()
