# WriterForcing: Generating more interesting story endings

Note
You need to have a GPU (and CUDA) to exucute the code. 

# Setup instructions

Create a conda environment: 

```
conda create -n writerForcing python=3.6
```

Activate the environment:

```
source activate writerForcing
```

Install a few required packages: 

```
conda install pytorch torchvision -c pytorch 

pip install torchtext 

pip install rake_nltk
```

Download the Glove embeddings from [here](https://nlp.stanford.edu/projects/glove/) and copy the file named "glove.6B.200d.txt" in the *data* folder.

## To train a model with "keyphrase addition", please run the following command:
```
python -m model.model_train_seq2seq --keyword_attention alpha_add
```

## To train a model with "keyphrase attention loss", please run the following command:
```
python -m model.model_train_seq2seq --keyword_attention att_sum_mse
```

## To train a model with "context concatenation", please run the following command:
```
python -m model.model_train_seq2seq --keyword_attention context_add
```

## To train a model with "coverage loss", please run the following command:
```
python -m model.model_train_seq2seq --keyword_attention att_sum_coverage
```

## Distinct Metric :
```
python calculate_metrics.py --file output_file_name_here.txt
``` 
