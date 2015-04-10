import sys
import numpy
import argparse 

from defgen_rev import train


def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    trainerr, validerr, testerr = train(saveto=params['model_name'],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        ctx_dim=params['embedding_dim'],
                                        dim=params['dim'][0],
                                        n_words=-1,
                                        decay_c=params['decay-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        maxlen=600,
                                        batch_size=16,
                                        valid_batch_size=16,
                                        validFreq=5000,
                                        sampleFreq=100,
                                        dispFreq=10,
                                        saveFreq=5000,
                                        dataset=params['data_file'],
                                        dictionary=params['dictionary_file'],
                                        use_dropout=True if params['use-dropout'][0] else False)
    return validerr

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'please specify model options: -m (model name), -da (data_file) , -dic (dictionary_file), -edim (embedding dimension)'
    else:
        options = {
        'dim_word': [256],
        'dim': [512],
        'n_layers': [1],
        'n-words': [30000], 
        'optimizer': ['adadelta'],
        'decay-c': [0.], 
        'use-dropout': [0],
        'learning-rate': [0.0001],
        'reload': [False]}
        print len(options)
        parser = argparse.ArgumentParser(description='parse model spec')
        parser.add_argument('-m','--model_name', type=str, help='a name for saving the model')
        parser.add_argument('-da','--data_file', type=str, help='a data file for training the model')
        parser.add_argument('-dic','--dictionary_file', type=str, help='a dictionary file for training the model')
        parser.add_argument('-edim','--embedding_dim', type=int, help='a dictionary file for training the model')
        options.update(vars(parser.parse_args()))
        print len(options), options.items() 

    main(0, options)
