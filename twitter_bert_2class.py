import numpy as np
import sqlite3
import tensorflow as tf
import transformers
from sklearn.metrics import accuracy_score

model_name = 'cl-tohoku/bert-base-japanese'
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

def read(DBPATH, TOPIC):
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    
    conn = sqlite3.connect(DBPATH)
    c = conn.cursor()
    cursor = c.execute(f"SELECT text,label FROM tw WHERE topic={TOPIC} AND (label='01000' OR label='00100') ORDER BY RANDOM()")
    for row in cursor:
        train_texts.append(row[0])
        if row[1] == '01000':
            train_labels.append(1)
        else:
            train_labels.append(0)
    conn.close()
    
    conn = sqlite3.connect(DBPATH)
    c = conn.cursor()
    cursor = c.execute(f"SELECT COUNT(text) FROM tw WHERE topic={TOPIC} AND (label='01000' OR label='00100')")
    for row in cursor:
        size = row[0]
    conn.close()
    
    conn = sqlite3.connect(DBPATH)
    c = conn.cursor()
    cursor = c.execute(f"SELECT MAX(LENGTH(text)) FROM tw WHERE topic={TOPIC} AND (label='01000' OR label='00100')")
    for row in cursor:
        max_length = row[0]
    conn.close()
    
    test_size = int(size * 0.3)
    test_texts = train_texts[-test_size:]
    test_labels = train_labels[-test_size:]
    
    print('train size: ', size - test_size, 'test size: ', test_size)
    return train_texts, train_labels, test_texts, test_labels, max_length

def to_features(texts, max_length):
    shape = (len(texts), max_length)
    input_ids = np.zeros(shape, dtype='int32')
    attention_mask = np.zeros(shape, dtype='int32')
    token_type_ids = np.zeros(shape, dtype='int32')
    for i, text in enumerate(texts):
        encoded_dict = tokenizer.encode_plus(text, max_length=max_length, pad_to_max_length=True, truncation=True)
        input_ids[i] = encoded_dict['input_ids']
        attention_mask[i] = encoded_dict['attention_mask']
        token_type_ids[i] = encoded_dict['token_type_ids']
    return [input_ids, attention_mask, token_type_ids]

def build_model(model_name, num_classes, max_length):
    input_shape = (max_length, )
    input_ids = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(input_shape, dtype=tf.int32)
    bert_model = transformers.TFBertModel.from_pretrained(model_name)
    last_hidden_state, pooler_output = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(pooler_output)
    model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model

num_classes = 2
batch_size = 64
epochs = 4

DBPATH = 'twitter/twitter.db'
TOPIC = '10020'
train_texts, train_labels, test_texts, test_labels, max_length = read(DBPATH, TOPIC)

x_train = to_features(train_texts, max_length)
y_train = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
model = build_model(model_name, num_classes=num_classes, max_length=max_length)

# train
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# predict
x_test = to_features(test_texts, max_length)
y_test = np.asarray(test_labels)
y_preda = model.predict(x_test)
y_pred = np.argmax(y_preda, axis=1)
print('Accuracy: %.5f' % accuracy_score(y_test, y_pred))