this project is based on the work of kashpop at https://github.com/kashpop/stancy
the purpose of the project is to enhance the model performance by adding information at the preprocessing.
the training and evaluation pipeline has a few parts:
3. finding and reading the data set on the machine. this is done at prep.py
2. preparing the data loader. this is done at bert_preprocessing.py
3. model training. there are a few files where this happens:
    base_bert.py - the vanilla BERT fine-tuning.
    bert_consistency.py - reproducing of the consistency architecture proposed by kashpop.
    models.py - holds the different architectures we used.
    consistency_ps_hp_tuning.py - consistency + part of speech model training
    bert_sentiment.py - consistency + sentiment model training
