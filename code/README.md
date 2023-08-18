# Latin_NER
This is the repo/folder for Latin_NER experiments
Assumes the "Latin-BERT" repository is cloned in the same folder

## preprocessing and train-test-split
EDA: [Title](new_minimal_EDA.ipynb)
Split the data: [Title](train_test_split2.ipynb)

## baselines

CRF: [Title](CRF_TEST_Herodotos.ipynb)
SpaCy: [Title](<Small SpaCy_Herodotos.ipynb>)

## LatinBERT
The training scripts:
[Title](latinbert_script_test.py) (no hypopt)

[Title](script_hypopt_latin_bert.py) (perform hypopt)
[Title](latinbert_hypopt_params.py) (retrain with best params)

Running trained model on the Herodotus test set using the Huggingface trainer API: [Title](Latin_BERT_error_script.ipynb)

Running trained model on the LASLA corpus using the LatinNerPipeline:
[Title](run_on_lasla.py)
[Title](run_on_lasla_separate_words.py)


LatinNerPipeline minimal example (slightly adjusted from the LASLA one): [Title](<pipeline demo.ipynb>) 