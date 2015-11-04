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
                                        n_words=params['n-words'],
                                        decay_c=params['decay-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0], 
                                        maxlen=100,
                                        batch_size=16,
                                        valid_batch_size=16,
                                        validFreq=5000,
                                        sampleFreq=100,
                                        dispFreq=10,
                                        saveFreq=5000,
                                        dataset=params['data_file'],
                                        dictionary=params['dictionary_file'],
                                        embeddings=params['embeddings'],
                                        feedforward=params['feedforward'],
                                        regress=params['regress'],
                                        use_target_as_input=params['use_target_as_input'],
                                        loss_type=params['loss_type'],
                                        use_mean=params['use_mean'],
                                        ranking_tol=0.1,
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
        'use_target_as_input': [True],
        'reload': [False]}

        print len(options)
        parser = argparse.ArgumentParser(description='parse model spec')
        parser.add_argument('-m','--model_name', type=str, help='a name for saving the model')
        parser.add_argument('-da','--data_file', type=str, help='a data file for training the model')
        parser.add_argument('-dic','--dictionary_file', type=str, help='a dictionary file for training the model')
        parser.add_argument('-embs','--embeddings', type=str, help='w2v embeddings (required if -ti flag true)')
        parser.add_argument('-edim','--embedding_dim', type=int, help='a dictionary file for training the model')
        parser.add_argument('-ff', '--feedforward', action='store_true',help='BOW model or RNN/LSTM?: -ff gives BOW')
        parser.add_argument('-rr', '--regress', action='store_true',help='just a linear map of added word embeddings')
        parser.add_argument('-ti', '--use-target-as-input', action='store_true',help='use pre-trained word embeddings to represent input words (uses embs from -embs flag)')
        parser.add_argument('--use-mean', action='store_true',help='RNN "output" is mean of all hidden states, not just final hidden state (makes little difference)')
        parser.add_argument('-l','--loss-type', type=str, default='cosine', help='loss type; "cosine" or "ranking"')
        options.update(vars(parser.parse_args()))
        print len(options), options.items() 

    main(0, options)
