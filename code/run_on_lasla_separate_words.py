
import os
from io import open
from conllu import parse_incr, TokenList



import pandas as pd

from tensor2tensor.data_generators import text_encoder

from transformers import BatchEncoding

class LatinTokenizer():
	def __init__(self, encoder):
		self.vocab={}
		self.reverseVocab={}
		self.encoder=encoder

		self.vocab["[PAD]"]=0
		self.vocab["[UNK]"]=1
		self.vocab["[CLS]"]=2
		self.vocab["[SEP]"]=3
		self.vocab["[MASK]"]=4
		self.model_max_length=256
		self.is_fast=False


		self.cls_token_id = self.vocab["[CLS]"]
		self.pad_token_id = self.vocab["[PAD]"]
		self.sep_token_id = self.vocab["[SEP]"]
        
		for key in self.encoder._subtoken_string_to_id:
			self.vocab[key]=self.encoder._subtoken_string_to_id[key]+5
			self.reverseVocab[self.encoder._subtoken_string_to_id[key]+5]=key


	def convert_tokens_to_ids(self, tokens):
		wp_tokens=[]
		for token in tokens:
			if token == "[PAD]":
				wp_tokens.append(0)
			elif token == "[UNK]":
				wp_tokens.append(1)
			elif token == "[CLS]":
				wp_tokens.append(2)
			elif token == "[SEP]":
				wp_tokens.append(3)
			elif token == "[MASK]":
				wp_tokens.append(4)

			else:
				wp_tokens.append(self.vocab[token])

		return wp_tokens

	def tokenize(self, text, split_on_tokens=True):
		if split_on_tokens:
			tokens = [token.lower() if token not in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] else token for token in text]
		else: 
			tokens = text.split()

		wp_tokens=[] #word-piece tokens
		check = []

		for n, token in enumerate(tokens):
			# print(token)

			if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
				wp_tokens.append(token)
				check.append(n)
			else:

				wp_toks=self.encoder.encode(token)

				for wp in wp_toks:
					wp_tokens.append(self.reverseVocab[wp+5])
					check.append(n)

		return wp_tokens, check
	
	def calculate_attention_masks(self, wp_tokens):
		attention_masks = []
		
		for token in wp_tokens:
			if token == self.pad_token_id:
				attention_masks.append(0)
			else:
				attention_masks.append(1)
				
		return attention_masks
	
	def pad(self, features, padding=True, max_length=256, pad_to_multiple_of="", return_tensors=True):
		# TODO
		batch_outputs = {}
		
		for i in range(len(features)):
			for key, value in features[i].items():
	
				if key in batch_outputs:
					batch_outputs[key].append(value)
	
				else:
					batch_outputs[key] = [value]

		for k, v in batch_outputs.items():
			batch_outputs[k] = torch.tensor([x for x in v])

		return BatchEncoding(batch_outputs)
	
	def pad_max_length_and_add_specials_tokens_also(self, tokens, wp_tokens):

		MAX_LENGTH = 256
		wp_tokens.insert(0, self.cls_token_id)
		tokens.insert(0, '[CLS]')
		wp_tokens.append(self.sep_token_id)
		tokens.append('[SEP]')
		
		if len(wp_tokens) > 256:
			wp_tokens = wp_tokens[:256]
		
		else:
			while len(wp_tokens) < 256:
				wp_tokens.append(self.pad_token_id)
				tokens.append('[PAD]')

		return tokens, wp_tokens
	
	def pad_max_length_and_add_specials(self, wp_tokens):

		MAX_LENGTH = 256
		wp_tokens.insert(0, self.cls_token_id)
		wp_tokens.append(self.sep_token_id)
		
		if len(wp_tokens) > 256:
			wp_tokens = wp_tokens[:256]
		
		else:
			while len(wp_tokens) < 256:
				wp_tokens.append(self.pad_token_id)

		return wp_tokens
	
	def decode_to_string(self, input_ids):
		tokens = [self.reverseVocab[x] for x in input_ids if x > 4]
		return "".join(tokens).replace('_', ' ')

	def save_pretrained(self, output_dir):
		pass



from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained('Herodotos_trained_lat_BERT_hypopt_params')
tokenizer = LatinTokenizer(text_encoder.SubwordTextEncoder('../latin-bert/models/subword_tokenizer_latin/latin.subword.encoder'))




def extend_clear_list(temp, fixed, item):
    temp.append(item)
    fixed.append(int(np.mean(temp)))
    temp.clear()


def aggregate_ents(orig_tokens, wp_tokens, check, labels):
    try:
        assert len(wp_tokens) == len(labels) == len(check)
    except AssertionError:
        print('lenght tokens labels are not equal')
        print(wp_tokens)
        print(check)
        
    fixed_labels = []
    
    temp_label = []
    
    for i in range(len(wp_tokens)):
        try:
            if check[i+1] != check[i] and len(temp_label) == 0:
                fixed_labels.append(labels[i])
            
            elif check[i+1] != check[i]:
                extend_clear_list(temp_label, fixed_labels, labels[i])
            else:
                temp_label.append(labels[i])
        except IndexError:
            if len(temp_label) == 0:
                fixed_labels.append(labels[i])
            
            elif len(temp_label) != 0:
                extend_clear_list(temp_label, fixed_labels, labels[i])
            
        
            
    try:        
        assert len(orig_tokens) == len(fixed_labels)
    except AssertionError:
        print('lenght of original tokens, aggregated predictions and labels are not equal')
        print(fixed_labels)
        print(check)
        print(orig_tokens)
    
    return fixed_labels



from transformers import Pipeline
from transformers.pipelines import TokenClassificationPipeline
import numpy as np
import re

def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

class LatinNerPipeline(TokenClassificationPipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "split_on_words" in kwargs:
            preprocess_kwargs["split_on_words"] = kwargs["split_on_words"]
        if "tokenizer" in kwargs:
            preprocess_kwargs["tokenizer"] = kwargs["tokenizer"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, split_on_words=False, tokenizer=tokenizer):
        test = {}
        if split_on_words:
            tokens = inputs
        else:
            tokens = inputs.split()

        wp_tokens, check = tokenizer.tokenize(tokens) #create wordpiece tokens
        token_ids = tokenizer.convert_tokens_to_ids(wp_tokens) #to ids
        test['inputs'] = inputs
        test['input_ids'] = [token_ids]
        test['attention_mask'] = [tokenizer.calculate_attention_masks(token_ids)]
        test['wp_tokens'] = wp_tokens
        test['check'] = check
        
        return test

    def _forward(self, test):
        model_inputs = {}
        model_inputs['input_ids'] = test['input_ids']
        model_inputs['attention_mask'] = test['attention_mask']
        model_inputs = BatchEncoding(model_inputs, tensor_type="pt")
        outputs = self.model(**model_inputs)
        test['outputs'] = outputs
        return test

    def postprocess(self, test):
        logits = test['outputs']["logits"] if isinstance(test['outputs'], dict) else test['outputs'][0]
        logits = test['outputs'].logits[0].detach().numpy()
        
        probabilities = [softmax(i) for i in logits]

        best_classes = [np.argmax(prob) for prob in probabilities]
#         score = [probabilities[best_class].item() for best_class in best_classes]
        logits = logits.tolist()
        agg_classes = aggregate_ents(test['inputs'], test['wp_tokens'], test['check'], best_classes)
        labels = [self.model.config.id2label[best_class] for best_class in agg_classes]
        
        test['logits'] = logits
        test['labels'] = labels
        
        return test


my_pipeline = LatinNerPipeline(model=model, tokenizer=tokenizer)

model2 = AutoModelForTokenClassification.from_pretrained('Herodotos_trained_lat_BERT_worked')
tokenizer2 = LatinTokenizer(text_encoder.SubwordTextEncoder('../latin-bert/models/subword_tokenizer_latin/latin.subword.encoder'))
my_pipeline2 = LatinNerPipeline(model=model2, tokenizer=tokenizer2)



def word2features(sent, i):
    word = sent[i]
    
    features = {
        'bias': 1.0, 
        # 'word.lower()': word.lower(), This was also commented out in the original notebook
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


# In[95]:


import joblib

filename = 'trained_CRF/CRF_without_words_itself_as_feature_Herodotos.sav'

crf = joblib.load(filename)


# In[96]:


def run_crf(model_input):
    feats = sent2features(model_input)
    assert(len(feats) == len(model_input))
    preds = crf.predict([feats])
    preds = sum(preds, [])
    return preds


# ## SpaCy

# In[97]:


# !pip install spacy
# !pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl

import spacy
nlp = spacy.load('la_core_web_lg', exclude=['normer', 'morphologizer', 'trainable_lemmatizer', 'parser', 'tagger', 'lemma_fixer'])
print(nlp.pipeline)


# In[98]:


from spacy.tokens import Doc

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
            spaces[-1] = False
            
        return Doc(self.vocab, words=words, spaces=spaces)
    
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


# In[99]:


from spacy.training import offsets_to_biluo_tags
def run_spacy(model_input):
    doc = nlp(model_input)
    
    
    predictions=[(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    # print(predictions)
    ents = offsets_to_biluo_tags(doc, predictions)
    ents = [ent.replace("U-", "B-").replace("L-", "I-").replace("PERSON", "PERS").replace("NORP", "GRP") for ent in ents]
    
    
    return ents


# ## Run and write

# In[100]:


def find_accepted_ids(sent):
    #list of dict as input
    #make a new_tokenlist with all the allowed ids
    #if a "pluribusque" shows up (id 7-8), followed by "pluribus" (id=7) and "que" (id=8)
    #we want to run the model on the sentence that only contains "pluribusque"
    accepted_ids = []
    for i in sent:
        if isinstance(i['id'], int) is True:
            accepted_ids.append(i['id'])
    return accepted_ids

def add_labels_to_tokenlist(accepted_ids, labels, label_name):
    assert len(accepted_ids) == len(labels)
    for i,j in list(zip(accepted_ids, labels)):
        for z in range(len(sent)):
            if sent[z]['id'] == i:
                sent[z][label_name] = j


# In[102]:


#main

input_dir = '../LASLALinkedLila/conllup'
output_dir = 'lasla_output_split_words'

conllu_files = [x for x in os.listdir(input_dir) if x.endswith('.conllup')]

for file in conllu_files:
    data_file = open(input_dir + '/' + file, "r", encoding="utf-8")
    destination_file = output_dir + '/' + file.replace('.conllup', '_with_ents') + '.conllup'
    if os.path.exists(destination_file):
        open(destination_file, 'w').close()
    
    sents = []
    
    for tokenlist in parse_incr(data_file, fields = ['id', 'form', 'lemma', 'upos', 
                                                     'xpos', 'feats', 'head', 'deprel', 
                                                     'deps', 'misc','lila:flcat', 
                                                     'lila:sentid', 'lila:line',
                                                    'lila:lemma', 'lila:token', 
                                                     'ent_label_BERT','ent_label_BERT_hypopt',
                                                    'ent_label_SpaCy', 'ent_label_CRF']):
        sents.append(tokenlist)
        
    for sent in sents:
    

        accepted_ids = find_accepted_ids(sent)
    
        #join the forms of the new tokenlist
        model_input = [token['form'].replace(' ', '') for token in sent if token['id'] in (accepted_ids)]
    
        #run model on this
        #return list with predicted entitylabels
        predict = my_pipeline([model_input], split_on_words=True)
        bert_hypopt_labels = predict[0]['labels']
        predict_bert = my_pipeline2([model_input], split_on_words=True)
        bert_labels = predict_bert[0]['labels']
        crf_labels = run_crf(model_input)
        spacy_labels = run_spacy(' '.join(model_input))
    
        #add each individual entity label to its corresponding token in the tokenlist
        try:
            add_labels_to_tokenlist(accepted_ids, bert_labels, 'ent_label_BERT')
            add_labels_to_tokenlist(accepted_ids, bert_hypopt_labels, 'ent_label_BERT_hypopt')
            add_labels_to_tokenlist(accepted_ids, spacy_labels, 'ent_label_SpaCy')
            add_labels_to_tokenlist(accepted_ids, crf_labels, 'ent_label_CRF')
        except AssertionError:
            print(f'length accepted_ids and labels are not equal in {file}, {sent}')
        
        #reserialize and write to file
        with open(destination_file, 'a', encoding='UTF-8') as f:
                f.write(sent.serialize())



# In[ ]:




