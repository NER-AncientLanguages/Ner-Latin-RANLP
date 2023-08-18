# Ner-Latin-RANLP
Repository for NER data for Latin

# Code
This is the folder for Latin_NER experiments
Assumes the "Latin-BERT" repository is cloned in the same folder.
Environment specifications are available in [Title](../../Gitlab/code_Latin_NER/environment/latin_ner_pipeline_env.yaml)

## preprocessing and train-test-split
[EDA](code/new_minimal_EDA.ipynb)
[train-test split](code/train_test_split2.ipynb)

## baselines

[CRF](code/CRF_TEST_Herodotos.ipynb)
[SpaCy](<code/Small SpaCy_Herodotos.ipynb>)

## LatinBERT
The training scripts:
[without hyptopt](code/latinbert_script_test.py) 

[perform hyptopt](code/script_hypopt_latin_bert.py) 
[retrain with best params](code/latinbert_hypopt_params.py)

Running trained model on the Herodotos test set using the Huggingface trainer API: [here](code/Latin_BERT_error_script.ipynb)

Running trained model on the LASLA corpus using the LatinNerPipeline:
[-que attached](code/run_on_lasla.py)
[-que separate](code/run_on_lasla_separate_words.py)


LatinNerPipeline minimal example (slightly adjusted from the LASLA one): [minimal example](<code/pipeline demo.ipynb>) 