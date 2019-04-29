#! /usr/bin/env python2.6.6
#coding=utf-8

import os
import logging
from gensim import corpora, models, matutils
from pprint import pprint

def saveDictonary(dictionary):
    token2id = dictionary.token2id
    dfs = dictionary.dfs
    token_info = {}
    for word in token2id:
        token_info[word] = dict(
            word = word,
            id = int(token2id[word]),
            freq = int(dfs[token2id[word]])
        )
    token_items = token_info.values()
    token_items = sorted(token_items, key = lambda x:x['id'])
    #pprint(token_items)
    with open('./corpus_dic.txt', 'w', encoding='utf8') as f:
        for item in token_items:
            #f.write("%d\t%s\t%d\n" % (item['id'],item['word'],item['freq']))
            f.write("%d\t%s\t%d\n" % (item['id'],item['word'],item['freq']))

def show2dCorpora(corpus):
    nodes = list(corpus)
    ax0 = [x[0][1] for x in nodes] # 绘制各个doc代表的点
    ax1 = [x[1][1] for x in nodes]
    # print(ax0)
    # print(ax1)
    # plt.plot(ax0,ax1,'o')
    # plt.show()

def showTopic(datapath):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    class MyCorpus(object):
        def __iter__(self):
            for line in open(datapath, encoding='utf8'):
                yield line.split()

    # 文本语料
    Corp = MyCorpus()
    with open('./corpus_txt.txt', 'w', encoding='utf8') as f:
      for text in Corp:
        f.write(' '.join(text)+'\n')

    # 生成字典
    dictionary = corpora.Dictionary(Corp)
    dictionary.save('./dictionary.dic')
    saveDictonary(dictionary);


    #数字语料
    corpus = [dictionary.doc2bow(doc) for doc in Corp]
    with open('./corpus_bow.txt', 'w', encoding='utf8') as f:
      f.write(str(corpus)+"\n")


    #tfidf加权  
    tfidf_model = models.TfidfModel(corpus)
    # print tfidf_model.dfsx  
    # print tfidf_model.idf  
    tfidf_model.save('./tfidf.mdl')

    corpus_tfidf = tfidf_model[corpus]
    with open('./corpus_tfidf.txt', 'w', encoding='utf8') as f:
      for doc in corpus_tfidf:
        f.write(str(doc)+"\n")

    #lda_model = models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=100,update_every=1,chunksize=50000,passes=1)
    #lda_model.save('./model.lda');
    
    #with open('./corpus_lda_topic.txt', 'wb') as f:
    #   for i in range(0,100):
    #     f.write(str(lda_model.get_topic_terms(i))+"\n")
    
    #corpus_lda = lda_model[corpus_tfidf]
    #with open('./corpus_lda.txt', 'wb') as f:
    #  for doc in corpus_lda:
    #    f.write(str(doc)+"\n")

    print('-------------------------------------------------1')
    lsi_model = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=100,chunksize=1, distributed=False)
    print('-------------------------------------------------2')
    lsi_model.save('./model.lsi');
    print('-------------------------------------------------3')

    with open('./corpus_lsi_topic.txt', 'w', encoding='utf8') as f:
      for i in range(0,100):
        f.write(str(lsi_model.print_topics(i,12))+"\n")
    
    corpus_lsi = lsi_model[corpus_tfidf]
    with open('./corpus_lsi.txt', 'w', encoding='utf8') as f:
      for doc in corpus_lsi:
        f.write(str(doc)+"\n")

    #lda_dns = matutils.corpus2dense(corpus_lda,50).T

    #with open('./corpus_lda_dense.txt', 'w') as f:
    #  for doc in lda_dns:
    #    f.write(" ".join([str(item) for item in doc])+"\n")

