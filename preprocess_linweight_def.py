import cPickle as pkl
import numpy
import sys

from collections import OrderedDict
from nltk.tokenize import wordpunct_tokenize


######
input_file = sys.argv[1]
embedding_file = sys.argv[2]
output_file = sys.argv[3]
dictionary_file = sys.argv[4]
if len(sys.argv) > 5:
    existing_dict = sys.argv[5]
else:
    existing_dict = False
######



def main():
    '''takes a word:definition dictionary and produces an \
        embedding:encoded-definition dictionary. \
        outputs new or updated encoding dictionary'''		

    with open(input_file, 'rb') as f:
        wn_defs = pkl.load(f)

    print 'Loading w2v...',
    with open(embedding_file, 'rb') as f:
        w2v = pkl.load(f)
    print 'Done'

    # build dictionary
    print 'Building a dictionary...',
    wordcounts = OrderedDict()
    n_defs = 0
    maxdefs = 0
    for kk, vv in wn_defs.iteritems():
        if kk in w2v:
            vec = w2v[kk]
        elif kk.lower() in w2v:
            vec = w2v[kk.lower()]
        else:
            continue
	if len(vv) > maxdefs:
	    maxdefs = len(vv)
        for dd in vv:
            n_defs += 1
            words = wordpunct_tokenize(dd.strip())
            for ww in words:
                if ww not in wordcounts:
                    wordcounts[ww] = 1
                else:
                    wordcounts[ww] += 1

	
	
    words = wordcounts.keys()
    counts = wordcounts.values()

    sorted_idx = numpy.argsort(counts)

    if existing_dict:
        with open(existing_dict) as inp:
	    worddict = cPickle.load(inp)
	maxval = max(worddict.values())
	counter = 0
	for idx, sidx in enumerate(sorted_idx[::-1]):
	    if not words[sidx] in worddict:
	        counter +=1
		worddict[words[sidx]] = maxval + counter
	    else:
	        continue
    else:
	worddict = OrderedDict()
	for idx, sidx in enumerate(sorted_idx[::-1]):
            worddict[words[sidx]] = idx+2

    with open(dictionary_file, 'wb') as f:
        pkl.dump(worddict, f)
    print 'Done'

    x = []
    y = []

    print 'Collection begins...'
    ii = 0
    for kk, vv in wn_defs.iteritems():
        if kk in w2v:
            vec = w2v[kk]
        elif kk.lower() in w2v:
            vec = w2v[kk.lower()]
        else:
            continue
        defno = 0
        for dd in vv:
            words = wordpunct_tokenize(dd.strip())
            seq = [worddict[w] for w in words]
	    for __ in range(maxdefs-defno):
                x.append(vec)
		y.append(seq)

            	ii += 1
            defno +=1
            if numpy.mod(ii, 1000):
                print ii,'/ approx',maxdefs*n_defs/2,','
    print 'Done'
	
    print 'Saving...',
    with open(output_file, 'wb') as f:
        pkl.dump(x,f)
	pkl.dump(y,f)
    print 'Done'


if __name__ == '__main__':
    main()
