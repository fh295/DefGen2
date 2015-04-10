import cPickle
import numpy as np
import string

'''create training / test data from wiktionary files'''
# possible exclusions: 
# if the definition contains the word (as a substring)
# if the definition is tagged as 'obsolete'
# if the keyword contains either whitespace or numeric characters

# start by only implementing the final one of these conditions

# key parameters ################
embeddings_vocab = 'w2v_vocab.txt'
raw_data_file = '/local/scratch/fh295/Wikipedia/enwikt-defs-latest-en.tsv'
output_file = 'Wiktionary_defs.pkl'
max_definitions = 100 # max number of sentences taken from the first paragraph of the article
####################################

def bracket_sweep(s, open_bracket='{{', close_bracket='}}'):
	cl = len(close_bracket)
	p1 = s.rfind(open_bracket)
	p2 = s.find(close_bracket, p1)
	return s.replace(s[p1:p2+cl],"")

def bracket_parse(s, open_bracket='{{', close_bracket='}}'):
	while s.count(open_bracket) == s.count(close_bracket) and s.count(open_bracket) > 0:
		s = bracket_sweep(s, open_bracket, close_bracket)
	return s

def unpack_sweep(s, open_bracket='{{l', close_bracket='}}',barrier='|'):
	cl = len(close_bracket)
	bl = len(barrier)
	p1 = s.rfind(open_bracket)
	p2 = s.find(close_bracket, p1)
	p3 = s[:p2].rfind(barrier)
	return s.replace(s[p1:p2+cl],s[p3+bl:p2])
		
def unpack_parse(s, open_bracket='{{l', close_bracket='}}',barrier='|'):
	while s.count(open_bracket) == s.count(close_bracket) and s.count(open_bracket) > 0:
		s = unpack_sweep(s)
	return s

def square_sweep(s, open_bracket='[[', close_bracket=']]',barrier='|'):
	ol = len(open_bracket)
	cl = len(close_bracket)
	bl = len(barrier)
	p1 = s.rfind(open_bracket)

	p2 = s.find(close_bracket, p1)
	p3 = s[:p2].find(barrier,p1)
	if p3 > -1:
		return s.replace(s[p1:p2+cl],s[p1+ol:p3])
	else:
		return s.replace(s[p1:p2+cl],s[p1+ol:p2])
		
def square_parse(s, open_bracket='[[', close_bracket=']]',barrier='|'):
	while s.count(open_bracket) == s.count(close_bracket) and s.count(open_bracket) > 0:
		s = square_sweep(s)
	return s

def strip_punct(s):
	return s.translate(string.maketrans("",""), string.punctuation)


def load_wikt(filename, vocab=None, single_words=True, obsoletes=True, excl_word_in_def=False, max_def=max_definitions):
    D = {}
    counter = 0
    inp =  open(filename)
    for l in inp:
        if counter % 100 == 0:
            print '%s word:def pairs processed' % (counter)
        word = l.split('\t')[1]
        defn = l.split('\t')[-1]
        if single_words and ' ' in word.strip():
            continue
        if vocab and not (word in vocab or word.lower() in vocab):
            continue
        if obsoletes and 'obsolete' in defn:
            continue
        if excl_word_in_def and word in defn:
            continue
        defn_c1 = unpack_parse(defn)
        defn_c2 = bracket_parse(defn_c1)
        defn_c3 = square_parse(defn_c2)
        defn_c4 = strip_punct(defn_c3)
        defn_c5 = defn_c4.strip()
        if word in D and len(D[word]) > max_def:
            continue
        elif word in D and len(defn_c5) > 1:
            D[word].append(defn_c5)
        elif len(defn_c5) > 1:
            D[word] = [defn_c5]
            counter +=1
    inp.close()
    return D


if __name__ == '__main__':
	print 'loading vocab'
	vocab = set(open(embeddings_vocab).read().split())
	print 'vocab loaded'
	D = load_wikt(raw_data_file, vocab=vocab)
	print 'dumping output'
	with open(output_file,'w') as out:
		cPickle.dump(D,out)


	


					
		


		
		
