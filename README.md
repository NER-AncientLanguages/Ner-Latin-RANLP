# Ner-Latin-RANLP
Repository for NER data for Latin

# Latin_Gold_data

This folder includes manual Named Entity Recognition (NER) annotation for three texts:
- Cicero, _In M. Antonium Oratio Philippica prima_ (3754 tokens, annotated by Margherita Fantoli)
- Juvenal, _Satirae_ 1-3 (4399 tokens, annotated by Evelien De Graaf and Margherita Fantoli)
- Tacitus, _Historiae_ 1 (11983 tokens; annotated by Evelien De Graaf)

The tokens are taken from the [LASLA corpus](https://www.lasla.uliege.be/cms/c_8570411/fr/lasla-textes-latins).
Every token is associated to its token and lemma URIs as found in the [LiLa Knowledge Base](https://lila-erc.eu/query/).

# Latin_Silver_Data

This folder contains the automatic NER annotation for the texts included in the portion of the LASLA corpus linked to the LiLa Knowledge Base, and includes the URIs of the linking. The model used for the annotation is found at [retrain with best params](https://github.com/NER-AncientLanguages/Ner-Latin-RANLP/blob/main/code/latinbert_hypopt_params.py).
See the following paper about the LiLa-LASLA linking:
[Linking the LASLA Corpus in the LiLa Knowledge Base of Interoperable Linguistic Resources for Latin](https://aclanthology.org/2022.ldl-1.4) (Fantoli et al., LDL 2022)

# Code
This is the folder for Latin_NER experiments, 
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
