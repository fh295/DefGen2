'''
Build a soft-attention-based image caption generator
'''
import theano
import theano.tensor as tensor

from defgen import *

#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

import cPickle as pkl
import numpy
import copy
import os

from scipy import optimize, stats
from collections import OrderedDict
from sklearn.cross_validation import KFold

import load_prepare_data


# all parameters
def init_params(options):
    params = OrderedDict()
    # embedding
    if not options['use_target_as_input']:
        params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    # encoder: LSTM
    if options.setdefault('feedforward', False):
        params['Wff'] = norm_weight(options['dim_word'], options['dim'])
    elif options.setdefault('regress', False):
        params['Wff'] = norm_weight(options['dim_word'], options['dim'])
    else:
        params = get_layer('lstm')[0](options, params, prefix='encoder',
                                      nin=options['dim_word'], dim=options['dim'])
    # readout
    if 'n_layers' in options:
        for lidx in xrange(1, options['n_layers']):
            params = get_layer('ff')[0](options, params, prefix='ff_out_%d'%lidx, nin=options['dim'], nout=options['dim'])

    # to output a word vector
    params = get_layer('ff')[0](options, params, prefix='ff_out', nin=options['dim'], nout=options['ctx_dim'])

    return params

# build a training model
def build_model(tparams, options):
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    if options['use_target_as_input']:
        x = tensor.tensor3('x', dtype='float32')
    else:
        x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    # context: #samples x dim
    ctx = tensor.matrix('ctx', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding
    if options['use_target_as_input']:
        emb = x
    else:
        emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    # decoder
    if options.setdefault('feedforward', False):
        proj_h = tensor.dot(emb, tparams['Wff'])
        proj_h = (proj_h * mask[:,:,None]).sum(axis=0)
        proj_h = proj_h / mask.sum(axis=0)[:,None]
    elif options.setdefault('regress', False):
        proj_h = (emb * mask[:,:,None]).sum(axis=0)
        proj_h = tensor.dot(proj_h, tparams['Wff'])
        proj_h = proj_h / mask.sum(axis=0)[:,None]
    else:
        proj = get_layer('lstm')[1](tparams, emb, options, 
                                    prefix='encoder', 
                                    mask=mask)
        proj_h = proj[0]
        if options['use_mean']:
            proj_h = (proj_h * mask[:,:,None]).sum(axis=0)
            proj_h = proj_h / mask.sum(axis=0)[:,None]
        else:
            proj_h = proj_h[-1]

    if 'n_layers' in options:
        for lidx in xrange(1, options['n_layers']):
            proj_h = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_out_%d'%lidx, activ='tanh')
    out = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_out', activ='linear')

    # cost
    if options['loss_type'] == 'cosine':
        out = out / tensor.sqrt((out ** 2).sum(1))[:,None]
        cost = 1. - (out * ctx).sum(1)
    elif options['loss_type'] == 'ranking':
        out = out / tensor.sqrt((out ** 2).sum(1))[:,None]
        rndidx = trng.permutation(n=ctx.shape[0])
        ctx_rnd = ctx[rndidx]
        cost = tensor.maximum(0., 1 - (out * ctx).sum(1) + (out * ctx_rnd).sum(1))
    else:
        raise Exception('Unknown loss function')

    return trng, use_noise, x, mask, ctx, cost

def build_fprop(tparams, options, trng, use_noise):
    # description string: #words x #samples
    if options['use_target_as_input']:
        x = tensor.tensor3('x', dtype='float32')
    else:
        x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding
    if options['use_target_as_input']:
        emb = x
    else:
        emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    # decoder
    if options.setdefault('feedforward', False):
        proj_h = tensor.dot(emb, tparams['Wff'])
        proj_h = (proj_h * mask[:,:,None]).sum(axis=0)
        proj_h = proj_h / mask.sum(axis=0)[:,None]
    elif options.setdefault('regress', False):
        proj_h = (emb * mask[:,:,None]).sum(axis=0)
        proj_h = tensor.dot(proj_h, tparams['Wff'])
        proj_h = proj_h / mask.sum(axis=0)[:,None]
    else:
        proj = get_layer('lstm')[1](tparams, emb, options, 
                                    prefix='encoder', 
                                    mask=mask)
        proj_h = proj[0]
        if 'use_mean' in options and options['use_mean'] or not 'use_mean' in options:
            proj_h = (proj_h * mask[:,:,None]).sum(axis=0)
            proj_h = proj_h / mask.sum(axis=0)[:,None]
        elif not options['use_mean']:
            proj_h = proj_h[-1]

    out = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_out', activ='linear')

    f_out = theano.function([x, mask], out, name='f_out')

    return f_out


def pred_probs(f_log_probs, prepare_data, data, iterator, 
               use_target_as_input=False, wv_embs=None, 
               verbose=False):
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 1)).astype('float32')

    n_done = 0

    for _, valid_index in iterator:
        x, mask, ctx = prepare_data([data[1][t] for t in valid_index], 
                                    [data[0][t] for t in valid_index])

        if use_target_as_input:
            shp = x.shape
            x = wv_embs[x.flatten()].reshape([shp[0], shp[1], wv_embs.shape[1]])

        pred_probs = f_log_probs(x,mask,ctx)
        probs[valid_index] = pred_probs[:,None]

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples computed'%(n_done,n_samples)

    return probs

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up)
    
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore')

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore')

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup)

    return f_grad_shared, f_update


def train(dim_word=100, # word vector dimensionality
          ctx_dim=512, # context vector dimensionality
          dim=1000, # the number of LSTM units
          n_layers=1, # output layer
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0., 
          lrate=0.01, 
          n_words=100000,
          maxlen=100, # maximum length of the description
          optimizer='rmsprop', 
          batch_size = 16,
          valid_batch_size = 16,
          embeddings='w2v.pkl',
          dataset='wn_w2v_defs',
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000, # save the parameters after every saveFreq updates
          sampleFreq=100, # generate some samples after every sampleFreq updates
          dictionary=None, # word dictionary
          use_dropout=False,
          use_target_as_input=False,
          loss_type='cosine', # 'cosine', 'ranking'
          ranking_tol=1.,
          feedforward=False,
          regress=False,
          use_mean=False,
          reload_=False):

    # Model options
    model_options = locals().copy()

    if dictionary:
        with open(dictionary, 'rb') as f:
            word_dict = pkl.load(f)
        word_idict = dict()
        for kk, vv in word_dict.iteritems():
            word_idict[vv] = kk

        if n_words > max(word_dict.values())+1 or n_words < 0:
            n_words = max(word_dict.values())+1
            model_options['n_words'] = n_words

    if use_target_as_input:
        assert dictionary, 'Dictionary must be provided'

        with open(embeddings, 'rb') as f:
            wv = pkl.load(f)
        wv_embs = numpy.zeros((n_words, len(wv.values()[0])), dtype='float32')
        for ii, vv in wv.iteritems():
            if ii in word_dict:
                wv_embs[word_dict[ii],:] = vv
        wv_embs = wv_embs.astype('float32')
        model_options['dim_word'] = wv_embs.shape[1]
    else:
        wv_embs = None

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl'%saveto, 'rb') as f:
            models_options = pkl.load(f)

    print 'Loading data'
    load_data, prepare_data = load_prepare_data.load_data, load_prepare_data.prepare_data
    train, valid, test = load_data(data_name=dataset, n_words=n_words, valid_portion=0.1)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
          x, mask, ctx, \
          cost = \
          build_model(tparams, model_options)

    # before any regularizer
    f_log_probs = theano.function([x, mask, ctx], -cost)

    cost = cost.mean()

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    f_cost = theano.function([x, mask, ctx], cost)

    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad = theano.function([x, mask, ctx], grads)

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, [x, mask, ctx], cost)

    print 'Optimization'

    if valid:
        kf_valid = KFold(len(valid[0]), n_folds=len(valid[0])/valid_batch_size, shuffle=True)
    if test:
        kf_test = KFold(len(test[0]), n_folds=len(test[0])/valid_batch_size, shuffle=True)

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        kf = KFold(len(train[0]), n_folds=len(train[0])/batch_size, shuffle=True)

        for _, train_index in kf:
            n_samples += train_index.shape[0]
            uidx += 1
            use_noise.set_value(1.)

            x, mask, ctx = prepare_data([train[1][t] for t in train_index], 
                                        [train[0][t] for t in train_index], 
                                        maxlen=maxlen)

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen
                continue

            if model_options['use_target_as_input']:
                shp = x.shape
                x = wv_embs[x.flatten()].reshape([shp[0], shp[1], wv_embs.shape[1]])

            cost = f_grad_shared(x, mask, ctx)
            f_update(lrate)

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                #import ipdb; ipdb.set_trace()

                if best_p != None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                with open('%s.pkl'%saveto, 'wb') as f:
                    pkl.dump(model_options, f)
                print 'Done'

            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = 0
                valid_err = 0
                test_err = 0
                #for _, tindex in kf:
                #    x, mask = prepare_data(train[0][train_index])
                #    train_err += (f_pred(x, mask) == train[1][tindex]).sum()
                #train_err = 1. - numpy.float32(train_err) / train[0].shape[0]

                #train_err = pred_error(f_pred, prepare_data, train, kf)
                if valid:
                    valid_err = -pred_probs(f_log_probs, prepare_data, valid, kf_valid, 
                                            use_target_as_input=use_target_as_input,
                                            wv_embs=wv_embs).mean()
                if test:
                    test_err = -pred_probs(f_log_probs, prepare_data, test, kf_test,
                                           use_target_as_input=use_target_as_input,
                                           wv_embs=wv_embs).mean()

                history_errs.append([valid_err, test_err])

                if uidx == 0 or valid_err <= numpy.array(history_errs)[:,0].min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience,0].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

        #print 'Epoch ', eidx, 'Update ', uidx, 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

        print 'Seen %d samples'%n_samples

        if estop:
            break

    if best_p is not None: 
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    train_err = 0
    valid_err = 0
    test_err = 0
    #train_err = pred_error(f_pred, prepare_data, train, kf)
    if valid:
        valid_err = -pred_probs(f_log_probs, prepare_data, valid, kf_valid)
    if test:
        test_err = -pred_probs(f_log_probs, prepare_data, test, kf_test)


    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, train_err=train_err, 
                valid_err=valid_err, test_err=test_err, history_errs=history_errs, 
                **params)

    return train_err, valid_err, test_err



if __name__ == '__main__':
    pass












    


