
import re
import html
import numpy as np
import spacy
from spacy import displacy
from spacy.tokens import Doc
from spacy.attrs import ID

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import models, initializers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, ReLU, LSTM, Bidirectional, Lambda, LayerNormalization, TimeDistributed, Activation
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate


REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
REPLACE_BR = re.compile("(</br>)")
REPLACE_DIV_ENT = re.compile('<div class=\"entities\"')
PH_MARK = ['<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">',
           '<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">',
           '<mark class="entity" style="background: #aa9cfcm ; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">']
MARK_CLOSE = '</mark>'
SPAN = '<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">'
SPAN_CLOSE = '</span>'


class Text(object):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.vocab = spacy.vocab.link_vectors_to_models(self.nlp.vocab)
        self.max_question_len = 35
        self.max_answer_len = 200
        self.question = ""
        self.text = ""
        self.doc = None
        self.possible_answers = []
        self.question_tokens = []
        self.possible_answer_tokens = []
        self.possible_answer_scores = []
        self.bilstm_attn_model = None
        self.embedding_shape = None


    def start_model(self):
        # setup embedding matrix from existing Spacy Vocabulary 
        embedding_index = {}
        for key, vector in self.nlp.vocab.vectors.items():
            row = self.nlp.vocab.vectors.find(key=key) 
            word = self.nlp.vocab.strings[key]
            embedding_index[word] = row
        embedding_matrix = self.nlp.vocab.vectors.data
        self.embedding_shape = embedding_matrix.shape

        # initialize model and load trained weights
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        self.bilstm_attn_model = bilstm_attn(self.embedding_shape[0], self.embedding_shape[1], embedding_matrix, self.max_question_len, self.max_answer_len)
        self.bilstm_attn_model.load_weights('tfqa_bilstm_attn_model_weights.h5')

       
    def tokenize(self, question, text):
        '''
        Tokenize question into ID's
        Determine possible answers - break text into 1, 2 and 3 sentences
        Tokenize possible answer candidates
        '''
        self.question = question
        self.text = text
        self.doc = self.nlp(text)

        # find answer candidates
        sent_list = list(self.doc.sents)
        num_sents = len(sent_list)
        for sent in sent_list:
            self.possible_answers.append(sent.text)
        for idx in range(num_sents - 1):
            self.possible_answers.append(self.possible_answers[idx] + self.possible_answers[idx + 1])
        for idx in range(num_sents - 2):
            self.possible_answers.append(self.possible_answers[idx] + self.possible_answers[idx + 1] + self.possible_answers[idx + 2])

        # tokenize potential answers
        self.possible_answer_tokens = np.zeros((len(self.possible_answers), self.max_answer_len))
        # for ans in self.possible_answers:
        for idx, doc in enumerate(self.nlp.pipe(self.possible_answers, disable=["tagger", "parser", "ner"])):
            ta = doc.to_array([ID])
            self.possible_answer_tokens[idx, 0: len(ta)] = ta if len(ta) < self.max_answer_len else ta[0: self.max_answer_len]
        self.possible_answer_tokens[self.possible_answer_tokens < 0] = 0
        self.possible_answer_tokens[self.possible_answer_tokens > self.embedding_shape[0]] = 0

        # tokenize question into ID's
        tokens = np.zeros(self.max_question_len)
        doc = list(self.nlp.pipe([self.question], disable=["tagger", "parser", "ner"]))[0]
        tq = doc.to_array([ID])
        tokens[0: len(tq)] = tq if len(tq) < self.max_question_len else tq[0: self.max_question_len]
        tokens[tokens < 0] = 0
        tokens[tokens > self.embedding_shape[0]] = 0

        self.question_tokens = np.zeros((len(self.possible_answers), self.max_question_len))
        for idx in range(len(self.possible_answers)):
            self.question_tokens[idx, :] = tokens


    def entity_display(self):
        '''
        Return entity display for text body corpus
        '''
        htmlstr = displacy.render(self.doc, style="ent")
        htmlstr = REPLACE_BR.sub(" ", htmlstr)
        return htmlstr


    def answer(self):
        '''
        Find best possible answer with trained bilstm attention model
        '''
        self.possible_answer_scores = self.bilstm_attn_model.predict([self.question_tokens, self.possible_answer_tokens])
        ind = np.argsort(self.possible_answer_scores.flatten())

        answer = '<div class="entities" style="line-height: 2.5; direction: ltr">'
        answer += '<ol>'
        answer += '<li>{} -- Score: {} </li>'.format(self.possible_answers[ind[-1]], self.possible_answer_scores[ind[-1]])
        answer += '<li>{} -- Score: {} </li>'.format(self.possible_answers[ind[-2]], self.possible_answer_scores[ind[-2]])
        answer += '<li>{} -- Score: {} </li>'.format(self.possible_answers[ind[-3]], self.possible_answer_scores[ind[-3]])
        answer += '</ol></div>'

        print("Possible Answers:")
        for ans in self.possible_answers:
            print(ans)
        print("QA Scores: {}". format(self.possible_answer_scores))

        # create html reply for web page display
        possibilities = '<div class="entities" style="line-height: 2.5; direction: ltr">'
        possibilities += '<ul>'
        for idx in range(len(self.possible_answers)):
            possibilities += '<li>{} -- Score: {} </li>'.format(self.possible_answers[idx], self.possible_answer_scores[idx])
        possibilities += '</ul></div>'

        return answer, possibilities


# bidirectional LSTM model with multihead (self) attention for both questions and answered concatenated thru a final linear layer with activation 
# implement (dot product) attention with scaling
class Attention():
    def __init__(self, dim, dropout=0.1):
        self.temperature = np.sqrt(dim)
        self.dropout = Dropout(dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,2]) / self.temperature)([q, k])
        if mask is not None:
            attn = Add()([attn, Lambda(lambda x: (-1e+10) * (1 - x))(mask)])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

# implement multi-head attention using dot product attention
class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, num_head=15, dim=100, num_key=64, num_val=64, dropout=0.1):
        self.num_head = num_head
        self.dim = dim
        self.num_key = num_key
        self.num_val = num_val
        self.dropout = dropout
        self.qlayer = Dense(num_head * num_key, use_bias=False)
        self.klayer = Dense(num_head * num_key, use_bias=False)
        self.vlayer = Dense(num_head * num_val, use_bias=False)
        self.attention = Attention(dim)
        self.layer_norm = LayerNormalization()
        self.w_o = TimeDistributed(Dense(dim))

    def __call__(self, q, k, v, mask=None):
        ql = self.qlayer(q)   # [batch_size, len_q, num_head * num_key]
        kl = self.klayer(k)
        vl = self.vlayer(v)

        def reshape1(x):
            s = tf.shape(x)   # [batch_size, len_q, num_head * num_key]
            x = tf.reshape(x, [s[0], s[1], self.num_head, self.num_key])
            x = tf.transpose(x, [2, 0, 1, 3])  
            x = tf.reshape(x, [-1, s[1], self.num_key])  # [num_head * batch_size, len_q, num_key]
            return x
        
        ql = Lambda(reshape1)(ql)
        kl = Lambda(reshape1)(kl)
        vl = Lambda(reshape1)(vl)

        if mask is not None:
            mask = Lambda(lambda x: K.repeat_elements(x, self.num_head, 0))(mask)
        head, attn = self.attention(ql, kl, vl, mask=mask)  
                
        def reshape2(x):
            s = tf.shape(x)   # [num_head * batch_size, len_v, num_val]
            x = tf.reshape(x, [self.num_head, -1, s[1], s[2]]) 
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, s[1], self.num_head * self.num_val])  # [batch_size, len_v, num_head * num_val]
            return x
        
        head = Lambda(reshape2)(head)

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)

        return self.layer_norm(outputs), attn

# Bidirectional LSTM with multihead attention (embedded matrix from spacy model)
def bilstm_attn(embedding_size, embedding_dim, embedding_matrix, question_dim, answer_dim, num_class=1, dropout=0.1):         
    input_q = Input(shape=(question_dim,), dtype="int32")          
    input_a = Input(shape=(answer_dim,), dtype="int32")
    xq = Embedding(embedding_size, embedding_dim, embeddings_initializer= tf.keras.initializers.Constant(embedding_matrix), input_length=question_dim, trainable=False)(input_q)   
    xa = Embedding(embedding_size, embedding_dim, embeddings_initializer= tf.keras.initializers.Constant(embedding_matrix), input_length=answer_dim, trainable=False)(input_a)  
    
    # Parallel BiLSTM layers with attention
    xq = Bidirectional(LSTM(question_dim, return_sequences=True))(xq)
    xq = Bidirectional(LSTM(question_dim, return_sequences=True))(xq)
    xa = Bidirectional(LSTM(answer_dim, return_sequences=True))(xa)
    xa = Bidirectional(LSTM(answer_dim, return_sequences=True))(xa)
    
    # Attention Layer
    xq, attn_q = MultiHeadAttention(num_head=15, num_key=question_dim, num_val=question_dim, dropout=dropout)(xq, xq, xq)
    avg1d_q = GlobalAveragePooling1D()(xq)
    max1d_q = GlobalMaxPooling1D()(xq)
    
    xa, attn_a = MultiHeadAttention(num_head=15, num_key=answer_dim, num_val=answer_dim, dropout=dropout)(xa, xa, xa)
    avg1d_a = GlobalAveragePooling1D()(xa)
    max1d_a = GlobalMaxPooling1D()(xa)
    
    x = Concatenate()([avg1d_q, max1d_q, avg1d_a, max1d_a])
    x = Dense(256, activation='relu')(x)
    x = Dense(num_class, activation='sigmoid')(x)

    model = Model(inputs=[input_q, input_a], outputs=x)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    # model.compile(tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True), 'binary_crossentropy', metrics=['accuracy'])
    return model

