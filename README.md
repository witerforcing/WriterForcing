# WriterForcing: Generating more interesting story endings

This repository has code for the published work at ACL Storytelling workshop 2019.
Please cite this paper if you use any part of this code : https://www.aclweb.org/anthology/W19-3413

Note:
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

## Story cloze evaluation
As mentioned in the paper we have used BERT for story cloze evaluation. The model can be downloaded [here](https://drive.google.com/open?id=12ArE22n0Fizh9DFZfeCIoqXbnxkAVMWd) and must be placed outside the git repo.

To install BERT follow the instructions in this [repo](https://github.com/huggingface/pytorch-transformers): 
After training your model with flag```--keyword_attention att_sum_mse_True```  the outputs would be written to ```att_sum_mse_True```, then run the evalutation script as follows

```
cd evaluate_attributes
python evaluate_story_cloze.py --model1 outputs/ie.txt --model2 ../att_sum_mse_True/../att_sum_mse_True/keyadd_outputs_storys_
```

## Distinct Metric :
```
cd outputs
python calculate_metrics.py --file output_file_name_here.txt
``` 

## Best Outputs
The best output of all the models compared in the paper are present in the ```outputs``` folder
