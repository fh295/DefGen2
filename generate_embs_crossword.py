import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import argparse

import numpy
import cPickle as pkl


import urllib2
import urllib

from defgen_rev import build_fprop, \
                       load_params, \
                       init_params, \
                       init_tparams, \
                       zipp

def combine(ranking_1, ranking_2, cutoff):
    candidates = ranking_2[:cutoff]
    new_indices = numpy.searchsorted(ranking_1, candidates)
    new_candidates = candidates[numpy.argsort(new_indices)]
    scores = [(c, ii + int(numpy.where(new_candidates==c)[0])) for ii, c in enumerate(candidates)]
    new_order = sorted(scores, key=lambda x:x[1])
    new_order_cands = numpy.array([x[0] for x in new_order])
    return new_order_cands




def main(model, 
         dictionary,
         embeddings):

    # load model model_options
    with open('%s.pkl'%model, 'rb') as f:
        model_options = pkl.load(f)

    with open(dictionary, 'rb') as f:
        worddict = pkl.load(f)
    worddict_r = dict()
    for k, v in worddict.iteritems():
        worddict_r[v] = k
    worddict_r[0] = '<eos>'

    print 'Loading skipgram vectors...',
    with open(embeddings, 'rb') as f:
        wv = pkl.load(f)
    wv_vectors = numpy.zeros((len(wv.keys()), wv.values()[0].shape[0]))
    wv_words = [None] * len(wv.keys())
    for ii, (kk, vv) in enumerate(wv.iteritems()):
        wv_vectors[ii,:] = vv
        wv_words[ii] = kk
    wv_vectors = wv_vectors / (numpy.sqrt((wv_vectors ** 2).sum(axis=1))[:,None])
    print 'Done'

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.), name='use_noise')

    params = init_params(model_options)

    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_prop = build_fprop(tparams, model_options, trng, use_noise)
    
    while True:
        wordin = raw_input('Type a description (case-probably-sensitive): ')

        words = wordin.strip().split()
        seq = [worddict[w] if w in worddict else 1 for w in words] + [0]
        seq_embs = numpy.array([wv[w] for w in words if w in wv])
        linemb = numpy.sum(seq_embs, axis=0)
        query_len = int(raw_input('Word length: '))
        form = raw_input('Form: ')
        if len(form) != query_len:
            print 'Word Length must be equal to length of form!'
            print 'Form should be like ??e??a? where ?s are unknown'
            query_len = int(raw_input('Word length: '))
            form = raw_input('Form: ')
        knowns = {}
        for ii in range(len(form)):
            if form[ii] != '?':
                knowns[ii] = form[ii]
        print knowns
        print 'Unknown words: ',
        for w in words:
            if w not in worddict:
                print w,
        print

        vec = f_prop(numpy.array(seq).reshape([len(seq),1]).astype('int64'),
                     numpy.ones((len(seq),1)).astype('float32'))
        vec = vec / numpy.sqrt((vec ** 2).sum(axis=1))[:,None]
        sims_rnn = (wv_vectors * vec).sum(1)
        sorted_idx_rnn = sims_rnn.argsort()[::-1]
        sims_w2v = (wv_vectors * linemb).sum(1)
        sorted_idx_w2v = sims_w2v.argsort()[::-1]
        query = urllib.urlencode([("rd", wordin.strip())])
        ret = urllib2.urlopen("http://api.datamuse.com/words?max=1000&"+query).read()
        wordlist = [s['word'] for s in eval(ret)]
        
        
        def match(word,word_len,knowndict):
            if len(word) != word_len:
                return False
            else:
                tomatch = len(knowndict)
                matches = len([i for i in knowndict if word[i] == knowndict[i]])
                if tomatch == matches:
                    return True
                else:
                    return False
        counter = 0
        print 'RNN candidates: '
        for ii, s in enumerate(sorted_idx_rnn):
            if counter <= 10:
                if match(wv_words[s], query_len, knowns):
                    print  ii, '(', sims_rnn[s],')-', wv_words[s]
                    counter +=1
            else:
                break
        print
        print 'w2v candidates: '
        counter = 0
        for ii, s in enumerate(sorted_idx_w2v):
            if counter <= 10:
                if match(wv_words[s], query_len, knowns):
                    print  ii, '(', sims_w2v[s],')-', wv_words[s]
                    counter +=1
            else:
                break

        print 'OneLook: '
        counter = 0
        for ii, s in enumerate(wordlist):
            if counter <= 10:
                if match(s, query_len, knowns):
                    print  ii, s
                    counter +=1
            else:
                break


        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str)
    parser.add_argument('-e','--embeddings',type=str)
    parser.add_argument('-dic','--dictionary',type=str)
    args = parser.parse_args()

    main(args.model, dictionary=args.dictionary,embeddings=args.embeddings)
