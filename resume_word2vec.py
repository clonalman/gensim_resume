#! /usr/bin/env python2.6.6
#coding=utf-8

import gensim

def saveWord2Vec(datapath):
    class MyCorpus(object):
        def __iter__(self):
            for line in open(datapath, encoding='utf8'):
                yield line.split()

    Corp = MyCorpus()
    with open('./corpus_wv_txt.txt', 'w', encoding='utf8') as f:
        for text in Corp:
            f.write(' '.join(text)+'\n')

    model = gensim.models.Word2Vec(Corp, workers=8, size=300)
    model.save('word2vec.model')
    print('save ok')

def testWord2Vec(word):
    model = gensim.models.Word2Vec.load('word2vec.model')
    print('"'+word+'"的特征向量是:')
    print(model[word])
    print('ok1')
    print(model.wv.vocab)
    print('ok2')

def listWord2Vec():
    model = gensim.models.Word2Vec.load('word2vec.model')
    with open('./corpus_wv_txt.words', 'wb') as f:
        for word in model.wv.vocab:
            f.write(str.encode(word+' '+ ' '.join([str(w) for w in list(model[word])]) +'\n'))


