#!/usr/bin/env python
# coding: utf-8

# In[10]:


# !pip install datasets
# !pip install transformers
# !pip install tensorflow
# !pip install tensor2tensor
# !pip install cltk
# !pip install seqeval
# !pip install wandb


# In[13]:


import pandas as pd
import json
from datasets import load_dataset
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from tensor2tensor.data_generators import text_encoder
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DefaultDataCollator
import numpy as np
from datasets import load_metric
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer


# In[20]:


from datasets import load_dataset

dataset = load_dataset('data/Latin_NER_json')

label2idx = {'O': 0, 
 'B-PERS': 1, 
 'I-PERS': 2, 
 'B-LOC': 3, 
 'I-LOC': 4, 
 'B-GRP': 5, 
 'I-GRP': 6}

idx2label = {value: key for key, value in label2idx.items()}


# In[24]:


#copied this class and function from the latin-BERT repo
#Wouters code
#made some adjustments

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

		for token in tokens:
			# print(token)

			if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
				wp_tokens.append(token)
			else:

				wp_toks=self.encoder.encode(token)

				for wp in wp_toks:
					wp_tokens.append(self.reverseVocab[wp+5])

		return wp_tokens
	
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
			batch_outputs[k] = torch.tensor([x.tolist() for x in v])

		return BatchEncoding(batch_outputs)
	
	def pad_max_length_and_add_specials_tokens_also(self, tokens, wp_tokens):

		MAX_LENGTH = 256
		wp_tokens.insert(0, self.cls_token_id)
		tokens.insert(0, '[CLS]')
		wp_tokens.append(self.sep_token_id)
		tokens.append('[SEP]')
		
		if len(wp_tokens) > 256:
			wp_tokens = wp_tokens[:256]
			tokens = tokens[:256]
		
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
tokenizer = LatinTokenizer(text_encoder.SubwordTextEncoder('../latin-bert/models/subword_tokenizer_latin/latin.subword.encoder'))


# In[18]:


import re


def tokenize_adjust_labels(all_samples_per_split):
	
	pretokenized_samples = all_samples_per_split["tokens"]
	tokenized_samples = tokenizer.tokenize(pretokenized_samples) #create wordpiece tokens
	token_ids = tokenizer.convert_tokens_to_ids(tokenized_samples) #to ids
    #path both the tokens and the the token_ids, as the tokenids are on subwords
	padded_tokenized_samples, padded_token_ids = tokenizer.pad_max_length_and_add_specials_tokens_also(tokenized_samples, token_ids) #pad and add special tokens

	all_samples_per_split['input_ids'] = padded_token_ids
    
	all_samples_per_split['attention_mask'] = tokenizer.calculate_attention_masks(padded_token_ids)
	all_samples_per_split['extra'] = padded_tokenized_samples

	#original
	orig_labels = all_samples_per_split['tags']


	# logic to adjust labels, 
	adjusted_labels = []
	label_idx = 0
	# print(len(pretokenized_samples))

	for token in padded_tokenized_samples:
		try:
            #The tokenizer always treats punctuation as a separate token
            #in most cases, this is not a problem as the punctuation is also seperately labeled in GWannotation, 
            #but there are a few exceptions
            #next statement catches those
            
			if token in ['[CLS]', '[SEP]', '[PAD]']:
				adjusted_labels.append(-100)
                
			elif re.match(r'\w+[\.\,]', pretokenized_samples[label_idx]):
				if token != '._' and token != ',_':
					adjusted_labels.append(orig_labels[label_idx])
				else:
					adjusted_labels.append(orig_labels[label_idx])
					label_idx += 1
		
			elif token.endswith('_'):
				adjusted_labels.append(orig_labels[label_idx])
				label_idx += 1

			else:
				adjusted_labels.append(orig_labels[label_idx])
		except IndexError:
			try :
				if token in ['[CLS]', '[SEP]', '[PAD]']:
					adjusted_labels.append(-100)
				else:
					adjusted_labels.append(orig_labels[label_idx])
			except IndexError:
				print('HERE')
				print(pretokenized_samples[:label_idx-1])
				print(orig_labels[:label_idx-1])
				print(token)
				print(list(zip(padded_tokenized_samples, adjusted_labels)))


	all_samples_per_split['labels'] = adjusted_labels

	try:
		assert len(adjusted_labels) == len(padded_tokenized_samples) == 256
	except AssertionError:
		print(all_samples_per_split)


	return all_samples_per_split


tokenized_dataset = dataset.map(tokenize_adjust_labels)
    


# In[19]:


tokenized_dataset


# In[21]:


import wandb
wandb.login(relogin=True)


# In[22]:


user = "myke-beersmans"
project = "Latin_ner_full_Herodotos"
display_name = "experiment-2023-05-30_18u40"

wandb.init(project=project, entity=user, name=display_name)


# In[23]:


metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [idx2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [idx2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    return flattened_results


# In[24]:


data_collator = DefaultDataCollator()


# In[25]:


#this should make everything fully reproducible
def model_init():
    model = AutoModelForTokenClassification.from_pretrained(
    "../latin-bert/models", num_labels=len(label2idx),
    label2id={0: 0, 
     1 : 1, 
     2 : 2, 
     3 : 3, 
     4 : 4, 
     5 : 5, 
     6 : 6},
    id2label={0:'O', 
     1 : 'B-PERS', 
     2 : 'I-PERS', 
     3 : 'B-LOC', 
     4 : 'I-LOC', 
     5 : 'B-GRP', 
     6 : 'I-GRP'}
     )
    
    model.config.start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


# In[26]:


# allright the training script works for real, something is wrong with the data
training_args = TrainingArguments(
    output_dir="./fine_tune_bert_output",
    evaluation_strategy="epoch",
    logging_strategy="epoch", 
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps = 1000,
    report_to="wandb",
    run_name = "Herodotos_dataset_3_epochs_30_may",
    save_strategy='no',
    warmup_ratio = 0.1,
    seed=123
)


trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


# In[27]:


trainer.evaluate()


# In[28]:


trainer.save_model('Herodotos_trained_lat_BERT_worked')


# In[29]:


wandb.finish()

