import cPickle as pkl
import numpy
import sys

from collections import OrderedDict
from nltk.tokenize import wordpunct_tokenize
from scipy.spatial.distance import cosine


######
input_file = sys.argv[1]
embedding_file = sys.argv[2]
output_file = sys.argv[3]
dictionary_file = sys.argv[4]
max_def_length = int(sys.argv[5])
if len(sys.argv) > 6:
    existing_dict = sys.argv[6]
else:
    existing_dict = False
######

def embedding_rank(word, def_list, emb_dict):
	defs_embs = []
		for defn in def_list:
			def_embs = []
			for w in defn:
				if w in emb_dict:
					def_embs.append(emb_dict[w])
				elif w.lower() in emb_dict:
					def_embs.append(emb_dict[w.lower()])
				else:
					continue
			if def_embs:	
			    def_emb = numpy.mean(numpy.array(def_embs), axis=0)
			else:
			    def_emb = numpy.zeros_like(emb_dict.values()[0])
			defs_embs.append(def_emb)
		try:
			word_emb = emb_dict[word]
		except:
			word_emb = emb_dict[word.lower()]
		dists = numpy.array([cosine(word_emb, emb) for emb in defs_embs])
		defs_sorted = numpy.argsort(dists) # smallest distance (most similar) first
		return [def_list[idx] for idx in defs_sorted]

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

    max_def = min(max_def_length,maxdefs)	
	
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

    with open(dictionary_file, 'w') as f:
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
	svv = embedding_rank(kk, vv, w2v) # sort the list of definitions in this case
        print 'original'
	print vv
	print 'sorted'
	print svv
        defno = 0
	for dd in svv:
            words = wordpunct_tokenize(dd.strip())
            seq = [worddict[w] for w in words]
	    for __ in range(max(0,max_def - defno)):
                x.append(vec)
                y.append(seq)

            	ii += 1
	    defno += 1
            if numpy.mod(ii, 1000):
                print ii,'/ approx',max_def*n_defs/2,','
    print 'Done'
	
    print 'Saving...',
    with open(output_file, 'w') as f:
        pkl.dump(x,f)
	pkl.dump(y,f)
    print 'Done'


if __name__ == '__main__':
    main()
