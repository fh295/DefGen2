'''
Build a soft-attention-based image caption generator
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy
import os

from scipy import optimize, stats
from collections import OrderedDict
from sklearn.cross_validation import KFold

import load_prepare_data


# my own softmax for Rop
def _softmax(x):
    e_x = tensor.exp(x - x.max(axis=1, keepdims=True)) 
    out = e_x / e_x.sum(axis=1, keepdims=True) 
    return out

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise, 
            state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
            state_before * 0.5)
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

# all parameters
def init_params(options):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    # context projection
    if options['n_layers'] > 1:
        for lidx in xrange(1,options['n_layers']):
            params = get_layer('ff')[0](options, params, prefix='ff_proj_%d'%lidx, nin=options['ctx_dim'], nout=options['ctx_dim'])
    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=options['ctx_dim'], nout=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=options['ctx_dim'], nout=options['dim'])
    # decoder: LSTM
    params = get_layer('lstm_cond')[0](options, params, prefix='decoder', 
                                       nin=options['dim_word'], dim=options['dim'], 
                                       dimctx=options['ctx_dim'])
    # readout
    # from LSTM
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], nout=options['dim_word'])
    # from context
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx', nin=options['ctx_dim'], nout=options['dim_word'])
    # from previous word
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev', nin=options['dim_word'], nout=options['dim_word'])
    # to output
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim_word'], nout=options['n_words'])

    return params

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive'%kk)
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'), 
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          }

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01):
    if nout == None:
        nout = nin
    if nout == nin:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)

def linear(x):
    return x

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'U')].shape[0]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = _slice(preact, 3, dim)

        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tensor.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h, c

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    rval, updates = theano.scan(_step, 
                                sequences=[mask, state_below],
                                outputs_info = [tensor.alloc(0., n_samples, dim),
                                                tensor.alloc(0., n_samples, dim)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval

# Conditional LSTM layer 
def param_init_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None):
    if nin == None:
        nin = options['dim']
    if dim == None:
        dim = options['dim']
    if dimctx == None:
        dimctx = options['dim']
    # input to LSTM
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W

    # LSTM to LSTM
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    # bias to LSTM
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx,dim*4)
    params[_p(prefix,'Wc')] = Wc

    return params

def lstm_cond_layer(tparams, state_below, options, prefix='lstm', 
                    mask=None, context=None, one_step=False, 
                    init_memory=None, init_state=None, 
                    **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_memory, 'previous memory must be provided'
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'U')].shape[0]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial/previous memory 
    if init_memory == None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # projected context 
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc')])

    # projected x
    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_, pctx_):

        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += pctx_

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tensor.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h, c

    if one_step:
        rval = _step(mask, state_below, init_state, init_memory, pctx_)
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=[mask, state_below],
                                    outputs_info = [init_state, init_memory],
                                    non_sequences=[pctx_],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps)
    return rval


# build a training model
def build_model(tparams, options, test=True):
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    # context: #samples x dim
    ctx = tensor.matrix('ctx', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    # initial state/cell
    init_state = get_layer('ff')[1](tparams, ctx, options, prefix='ff_state', activ='tanh')
    init_memory = get_layer('ff')[1](tparams, ctx, options, prefix='ff_memory', activ='tanh')
    # context project
    ctx_p = ctx
    if options['n_layers'] > 1:
        for lidx in xrange(1,options['n_layers']):
            ctx_p = get_layer('ff')[1](tparams, ctx_p, options, prefix='ff_proj_%d'%lidx, activ='tanh')
    # decoder
    proj = get_layer('lstm_cond')[1](tparams, emb, options, 
                                     prefix='decoder', 
                                     mask=mask, context=ctx_p, 
                                     one_step=False, 
                                     init_state=init_state,
                                     init_memory=init_memory)
    proj_h = proj[0]
    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctx_p, options, prefix='ff_logit_ctx', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options, prefix='ff_logit_prev', activ='linear')
    logit = tensor.tanh(logit_lstm + logit_ctx[None,:,:] + logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = _softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))
    # cost
    x_flat = x.flatten()
    if test:
        cost = -tensor.log(probs[tensor.arange(x_flat.shape[0]), x_flat])
    else:
        cost = -tensor.log(probs[tensor.arange(x_flat.shape[0]), x_flat]+1e-8)
    cost = cost.reshape([x.shape[0], x.shape[1]])
    cost = (cost * mask).sum(0)
    #cost = cost.mean()

    return trng, use_noise, x, mask, ctx, cost

# build a sampler
def build_sampler(tparams, options, trng):
    # context: 1 x dim
    ctx = tensor.matrix('ctx_sampler', dtype='float32')
    ctx_p = ctx
    if options['n_layers'] > 1:
        for lidx in xrange(1,options['n_layers']):
            ctx_p = get_layer('ff')[1](tparams, ctx_p, options, prefix='ff_proj_%d'%lidx, activ='tanh')


    # initial state/cell
    init_state = get_layer('ff')[1](tparams, ctx_p, options, prefix='ff_state', activ='tanh')
    init_memory = get_layer('ff')[1](tparams, ctx_p, options, prefix='ff_memory', activ='tanh')

    print 'Building f_init...',
    f_init = theano.function([ctx], [init_state, init_memory], name='f_init')
    print 'Done'

    # x: 1 x 1
    x = tensor.vector('x_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    init_memory = tensor.matrix('init_memory', dtype='float32')

    # if it's the first word, emb should be all zero
    emb = tensor.switch(x[:,None] < 0, tensor.alloc(0., x.shape[0], tparams['Wemb'].shape[1]), 
                        tparams['Wemb'][x])
    # context project
    proj = get_layer('lstm_cond')[1](tparams, emb, options, 
                                     prefix='decoder', 
                                     mask=None, context=ctx_p, 
                                     one_step=True, 
                                     init_state=init_state,
                                     init_memory=init_memory)
    next_state, next_memory = proj[0], proj[1]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options, prefix='ff_logit_lstm', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctx_p, options, prefix='ff_logit_ctx', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options, prefix='ff_logit_prev', activ='linear')
    logit = tensor.tanh(logit_lstm + logit_ctx + logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    f_next = theano.function([x, ctx, init_state, init_memory], [next_probs, next_sample, next_state, next_memory], name='f_next')

    return f_init, f_next

# build reverser
def build_reverser(x, mask, ctx, cost):
    # gradient of the cost w.r.t. to ctx
    ctx_grad = tensor.grad(cost, wrt=ctx)
    f_ctx_grad = theano.function([x, mask, ctx], ctx_grad)

    return f_ctx_grad

# build hessian p
def build_hess_p(x, mask, ctx, cost):
    p = tensor.matrix(name='p', dtype='float32')
    ctx_grad = tensor.grad(cost, ctx)
    ctx_hess_p = tensor.Rop(ctx_grad, ctx, p)
    f_ctx_hess_p = theano.function([x, mask, ctx, p], ctx_hess_p)

    return f_ctx_hess_p

# infer the word vector given a definition
def infer_ctx(options, seq, f_cost, f_ctx_grad, init_ctx = None, f_hess_p = None, maxiter=100):
    if init_ctx == None:
        init_ctx = 1e-3 * numpy.random.randn(1, options['ctx_dim']).astype('float32')
    x, mask, ctx0 = prepare_data([seq], init_ctx)

    def _g(ctx):
        return f_ctx_grad(x, mask, ctx.reshape([1, ctx.shape[0]]).astype('float32')).reshape([ctx.shape[0]])
    def _c(ctx):
        return f_cost(x, mask, ctx.reshape([1, ctx.shape[0]]).astype('float32'))
    def _hp(ctx, p):
        if f_hess_p:
            return f_hess_p(x, mask, ctx.reshape([1, ctx.shape[0]]), p.reshape([1, p.shape[0]])).astype('float32')
        else:
            return None
    def _cb(ctx):
        cc = f_cost(x, mask, ctx.reshape([1, ctx.shape[0]]).astype('float32'))
        print 'Current cost: ', cc

    if f_hess_p:
        ctx_opt = optimize.fmin_ncg(_c, ctx0[0,:], fprime=_g, fhess_p=_hp, callback=None, maxiter=maxiter)
    else:
        ctx_opt = optimize.fmin_bfgs(_c, ctx0[0,:], fprime=_g, callback=None, maxiter=maxiter)

    return ctx_opt

# generate sample
def gen_sample(tparams, f_init, f_next, ctx, 
               options, trng=None, k=1, maxlen=30, stochastic=False,
               allow_unk=True):
    if len(ctx.shape) == 1:
        ctx = ctx.reshape([1, ctx.shape[0]])
    ctx0 = ctx

    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    hyp_memories = []

    next_state, next_memory = f_init(ctx)
    next_w = -1 * numpy.ones((live_k,)).astype('int64')

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        next_p, next_w, next_state, next_memory = f_next(next_w, ctx, next_state, next_memory)

        if stochastic:
            sample.append(next_w[0])
            sample_score -= numpy.log(next_p[0,next_w[0]])
            if next_w[0] == 0:
                break
        else:
            logp = numpy.log(next_p)
            if not allow_unk:
                logp[:,1] = -numpy.Inf
            cand_scores = hyp_scores[:,None] - logp
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_memories = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[ti])
                new_hyp_states.append(copy.copy(next_state[ti]))
                new_hyp_memories.append(copy.copy(next_memory[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            hyp_memories = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    hyp_memories.append(new_hyp_memories[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            next_memory = numpy.array(hyp_memories)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score

def pred_probs(f_log_probs, prepare_data, data, iterator, verbose=False):
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 1)).astype('float32')

    n_done = 0

    for _, valid_index in iterator:
        x, mask, ctx = prepare_data([data[1][t] for t in valid_index], 
                                    [data[0][t] for t in valid_index])
        pred_probs = f_log_probs(x,mask,ctx)
        probs[valid_index] = pred_probs[:,None]

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples computed'%(n_done,n_samples)

    return probs

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update

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
          n_layers=1,
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
          dataset='wn_w2v_defs',
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000, # save the parameters after every saveFreq updates
          sampleFreq=100, # generate some samples after every sampleFreq updates
          dictionary=None, # word dictionary
          use_dropout=False,
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
          build_model(tparams, model_options, test=False)

    print 'Buliding sampler'
    f_init, f_next = build_sampler(tparams, model_options, trng)

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

            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                x_s, mask_s, ctx_s = prepare_data([train[1][t] for t in xrange(10)], 
                                                  [train[0][t] for t in xrange(10)])
                for jj in xrange(10):
                    sample, score = gen_sample(tparams, f_init, f_next, ctx_s[jj], model_options,
                                        trng=trng, k=1, maxlen=30, stochastic=True)
                    print 'Truth ',jj,': ',
                    for vv in x_s[:,jj]:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            print word_idict[vv], 
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    for vv in sample:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            print word_idict[vv], 
                        else:
                            print 'UNK',
                    print

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
                    valid_err = -pred_probs(f_log_probs, prepare_data, valid, kf_valid).mean()
                if test:
                    test_err = -pred_probs(f_log_probs, prepare_data, test, kf_test).mean()

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












    


