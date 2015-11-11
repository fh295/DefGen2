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

def main(model, 
         dictionary,
         embeddings, shortlist=None):

    if shortlist:
        shortlist = set(open(shortlist).read().split())
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


    wv_rdict = {}
    for ii, (kk, vv) in enumerate(wv.iteritems()):
        wv_vectors[ii,:] = vv
        wv_words[ii] = kk
        wv_rdict[kk] = ii
    wv_vectors_normed = wv_vectors / (numpy.sqrt((wv_vectors ** 2).sum(axis=1))[:,None])
    wv_normed_dict = {wv_words[ii]:vv for ii,vv in enumerate(wv_vectors)}
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
        length = raw_input('Answer length: ')
        try:
            length = int(length)
        except:
            length = None
        def is_length(word,length):
           if length:
               return length == len(word)
           else:
               return True
	if model_options['use_target_as_input']:
	    print 'w2v input used'
            seq = numpy.array([wv_rdict[w] if w in wv_rdict else 1 for w in words] + [0])
            shp = seq.shape
	    seq = wv_vectors[seq.flatten()].reshape([shp[0], 1, wv_vectors.shape[1]]).astype('float32')
            vec = f_prop(seq, numpy.ones((seq.shape[0],1)).astype('float32'))[0][None,:]
	else:
	    print 'w2v input not used'
            seq = numpy.array([worddict[w] if w in worddict else 1 for w in words] + [0])
            vec = f_prop(numpy.array(seq).reshape([len(seq),1]).astype('int64'), \
                         numpy.ones((len(seq),1)).astype('float32'))
        seq_embs = numpy.array([wv_normed_dict[w] for w in words if w in wv_normed_dict])
        linemb = numpy.sum(seq_embs, axis=0)
        print 'Unknown words: ',
        for w in words:
            if w not in worddict:
                print w,
        print
        vec = vec / numpy.sqrt((vec ** 2).sum(axis=1))[:,None]
        sims_rnn = (wv_vectors_normed * vec).sum(1)
        sorted_idx_rnn = sims_rnn.argsort()[::-1]
        sims_w2v = (wv_vectors_normed * linemb).sum(1)
        sorted_idx_w2v = sims_w2v.argsort()[::-1]
        query = urllib.urlencode([("rd", wordin.strip())])
        ret = urllib2.urlopen("http://api.datamuse.com/words?max=1000&"+query).read()
        wordlist = [s['word'] for s in eval(ret)]
	print 'model used:', model
        if shortlist:
            sl = shortlist
        else:
            sl = worddict
        print 'RNN candidates: '
        counter = 0
        for ii, s in enumerate(sorted_idx_rnn):
            if wv_words[s] in sl and is_length(wv_words[s],length):
                print '', counter +1, '(', sims_rnn[s],')-', wv_words[s]
                counter +=1
            if counter > 10:
                break
        print
        counter = 0
        print 'w2v candidates: '
        for ii, s in enumerate(sorted_idx_w2v):
            if wv_words[s] in sl and is_length(wv_words[s],length):
                print '', counter +1, '(', sims_w2v[s],')-', wv_words[s]
                counter +=1
            if counter > 10:
                break
        print
        print 'OneLook candidates: '
        counter = 0
        for ii, s in enumerate(wordlist):
            if s in sl  and is_length(s,length):
                print '', counter+1, s
                counter +=1
            if counter > 10:
                break
        print



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str)
    parser.add_argument('-e','--embeddings',type=str)
    parser.add_argument('-dic','--dictionary',type=str)
    parser.add_argument('-sl','--output_shortlist',type=str)
    args = parser.parse_args()
    main(args.model, dictionary=args.dictionary,embeddings=args.embeddings, shortlist=args.output_shortlist)
