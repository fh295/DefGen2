import sys
import cPickle

dict1 = sys.argv[1]
dict2 = sys.argv[2]
dict3 = sys.argv[3]
outfile = sys.argv[4]

with open(dict1) as IN:
    D1 = cPickle.load(IN)

with open(dict2) as IN:
    D2 = cPickle.load(IN)

with open(dict3) as IN:
    D3 = cPickle.load(IN)

def merge_dicts(A,B):
    D = {}
    for a1,a2 in A.iteritems():
        if a1 in B:
            D[a1] = a2 + B[a1]
        elif a1.lower() in B:
            D[a1] = a2 + B[a1.lower()]
        else:
            D[a1] = a2
    for b1,b2 in B.iteritems():
        if b1.lower() in A and not b1.lower() in D:
            D[b1.lower()] = b2 + A[b1.lower()]
        elif not b1 in D:
            D[b1] = b2
    return D

if __name__ == '__main__':
    M1 = merge_dicts(D1,D2)
    M2 = merge_dicts(M1,D3)
    with open(outfile,'w') as out:
        cPickle.dump(M2,out)        
            
