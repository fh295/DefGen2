THEANO_FLAGS='floatX=float32' python preprocess_embeddingweight_def.py WN_definition.pkl D_w2v_300_1bn.pkl WN_data_T4.pkl WN_dict_T4.pkl 10

THEANO_FLAGS='floatX=float32' python preprocess_embeddingweight_def.py Wiktionary_defs.pkl D_w2v_300_1bn.pkl Wiktionary_data_T4.pkl Wiktionary_dict_T4.pkl 10

THEANO_FLAGS='floatX=float32' python preprocess_embeddingweight_def.py Wiktionary_defs.pkl D_w2v_300_1bn.pkl Wiki_data_T4.pkl Wiki_dict_T4.pkl 10

THEANO_FLAGS='floatX=float32' python preprocess_embeddingweight_def.py wiki_wikt_wn_merged_definitions.pkl D_w2v_300_1bn.pkl Merged_data_T4.pkl Merged_dict_T4.pkl 15
