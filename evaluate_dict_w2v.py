import numpy

from defgen_bs import train

def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    trainerr, validerr, testerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        ctx_dim=300,
                                        dim=params['dim'][0],
                                        n_words=-1,
                                        decay_c=params['decay-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        maxlen=600,
                                        batch_size=16,
                                        valid_batch_size=16,
                                        validFreq=1000,
                                        sampleFreq=100,
                                        dispFreq=10,
                                        saveFreq=1000,
                                        dictionary='/home/fh295/Documents/Deep_learning_Bengio/defgen/Dictionary_files/Wiki_dict_T4.pkl',
                                        use_dropout=True if params['use-dropout'][0] else False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['wiki_T4.npz'],
        'dim_word': [256],
        'dim': [1024],
        'n-words': [30000], 
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'use-dropout': [0],
        'learning-rate': [0.0001],
        'reload': [False]})

