import cPickle as pkl
import numpy

def prepare_data(seqs, contexts, maxlen=None):
    lengths = [len(s) for s in seqs]

    if maxlen != None:
        new_seqs = []
        new_contexts = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, contexts):
            if l < maxlen:
                new_seqs.append(s)
                new_contexts.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        contexts = new_contexts
        seqs = new_seqs

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, contexts


def load_data(dataset,n_words=20000, valid_portion=0.1):
    with open(dataset, 'rb') as f:
        x = pkl.load(f)
        y = pkl.load(f)

    n_samples = len(x)
    rndidx = numpy.random.permutation(n_samples)

    n_valid = numpy.round(n_samples * valid_portion)

    def remove_unk(v):
        return [[1 if w >= n_words else w for w in sen] for sen in v]

    def normalize(v):
        return v / numpy.sqrt(numpy.sum(v**2))

    x_val = [normalize(x[ii].astype('float32')) for ii in rndidx[-n_valid:]]
    y_val = remove_unk([y[ii] for ii in rndidx[-n_valid:]])
	
    x = [normalize(x[ii].astype('float32')) for ii in rndidx[:-n_valid]]
    y = remove_unk([y[ii] for ii in rndidx[:-n_valid]])

    return (x,y), (x_val,y_val), None

