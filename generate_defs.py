"""
Print a definition from a given word in the command line \
using a trained defgen model
"""

import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import argparse

import numpy
import cPickle as pkl

from defgen import build_sampler, gen_sample, \
                   load_params, \
                   init_params, \
                   init_tparams, \
                   zipp

def main(model, 
         dictionary='/home/fh295/Documents/Deep_learning_Bengio/defgen/Dictionary_files/Wiki_dict_T4.pkl',
         w2v='/home/fh295/Documents/Deep_learning_Bengio/defgen/D_w2v_300_1bn.pkl'):

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
    with open(w2v, 'rb') as f:
        wv = pkl.load(f)
    print 'Done'

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.), name='use_noise')

    params = init_params(model_options)

    params = load_params(model, params)
    tparams = init_tparams(params)

    # word index
    f_init, f_next = build_sampler(tparams, model_options, trng)
    
    while True:
        wordin = raw_input('Type a word (case-probably-sensitive): ')
        wordin = wordin.strip()
        sto = raw_input('Stochastic (y/n)? ')
        stochastic = True
        if sto == 'n':
            stochastic = False
        n_samples = int(raw_input('How many samples? '))

        if wordin in worddict or wordin.lower() in worddict:
            print 'This word was in the training set'

        if wordin not in wv and wordin.lower() not in wv:
            print 'Unknown word'
            continue

        if wordin in wv:
            vv = wv[wordin]
        if wordin.lower() in wv:
            vv = wv[wordin.lower()]

        if stochastic:
            samples = []
            scores = []
            
            for ii in xrange(n_samples):
                sample, score = gen_sample(tparams, f_init, f_next, 
                                           vv.astype('float32'), model_options,
                                           trng=trng, k=1, maxlen=10, stochastic=stochastic)
                samples.append(sample)
                scores.append(score)
        else:
            samples, scores = gen_sample(tparams, f_init, f_next, 
                                       vv.astype('float32'), model_options,
                                       trng=trng, k=n_samples, maxlen=10, stochastic=stochastic)

        sorted_idx = numpy.argsort(numpy.array(scores))

        print 'Generated samples: '
        for ii, s in enumerate(sorted_idx):
            print '', ii, '(', scores[s],')-', [worddict_r[w] if w in worddict_r else 'UNK' for w in samples[ii]]
        print


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)

    args = parser.parse_args()

    main(args.model)
