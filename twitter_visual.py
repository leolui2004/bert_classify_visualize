import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import sentencepiece as spm
import numpy as np
import pandas as pd
import sqlite3
from keras_bert import load_trained_model_from_checkpoint

def read(DBPATH, TOPIC):
    txt_list = []
    
    conn = sqlite3.connect(DBPATH)
    c = conn.cursor()
    cursor = c.execute(f"SELECT text,label FROM tw WHERE topic={TOPIC} AND (label='01000' OR label='00100') ORDER BY RANDOM()")
    for row in cursor:
        text = str(row[0]).replace('\n', '。').replace('！', '。').replace('　', '。').replace('、', '。')
        txt_split = text.split('。')
        for i in txt_split:
            if len(i) > 10:
                txt_list.append(i)
    conn.close()
    
    return txt_list

DIR = 'nlp/'
DBPATH = f'{DIR}twitter/twitter.db'
TOPIC = '10020'
txt_list = read(DBPATH, TOPIC)

bert_dir = f'{DIR}spm/'
config_path = f'{bert_dir}bert_config.json'
checkpoint_path = f'{bert_dir}model.ckpt-1400000'

bert = load_trained_model_from_checkpoint(config_path, checkpoint_path)
spp = spm.SentencePieceProcessor()
spp.Load(f'{DIR}spm/wiki-ja.model')

def vectorize(texts):
    maxlen = 200
    common_seg_input = np.zeros((len(texts), maxlen), dtype = np.float32)
    matrix = np.zeros((len(texts), maxlen), dtype = np.float32)
    for i, text in enumerate(texts):
        tok = [w for w in spp.encode_as_pieces(text.replace(' ', ''))]
        if tok == [] or len(tok) > maxlen:
            print('skip processing', tok)
        else:
            tokens = []
            tokens.append('[CLS]')
            tokens.extend(tok)
            tokens.append('[SEP]')
            for t, token in enumerate(tokens):
                try:
                    matrix[i, t] = spp.piece_to_id(token)
                except:
                    print(token+'is unknown')
                    matrix[i, t] = spp.piece_to_id('<unk>')
    return bert.predict([matrix, common_seg_input])[:,0]

vector_list = []
sentence_list= []
for i in txt_list:
    i_replace = i.replace('。', '')
    vector_list.append(vectorize([i_replace]))
    sentence_list.append(i_replace)

pd.DataFrame(np.vstack(vector_list)).to_csv(f'{DIR}twitter/twitter_vector.tsv', sep='\t', index=False, header=None)
pd.DataFrame(sentence_list).to_csv(f'{DIR}twitter/twitter_sentence.tsv', index=False, header=None)