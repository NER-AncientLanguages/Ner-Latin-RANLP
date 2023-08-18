

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
			tokens = [token.lower() for token in text.split()]

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
            
            
#     fixed_labels.append(labels[len(wp_tokens)-1])
            
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

    def preprocess(self, inputs, tokenizer=tokenizer, split_on_words=False):
        test = {}
        if split_on_words:
            tokens = inputs
        else:
            tokens = [token.lower() for token in inputs.split()]

        wp_tokens, check = tokenizer.tokenize(tokens) #create wordpiece tokens
        token_ids = tokenizer.convert_tokens_to_ids(wp_tokens) #to ids
        test['inputs'] = tokens
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