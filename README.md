<div align="center">
<h1>
LLM_NER_MultiNERD
</h1>

<center>
<img src="https://github.com/medxiaorudan/LLM_NER_MultiNERD/blob/main/images/displacy.PNG" width="700" > 
</center>

[🤗 MultiNERD Dataset](https://huggingface.co/models?library=span-marker) |
[🛠️ bert-base-cased model](https://huggingface.co/bert-base-cased) |
[🛠️ xlnet-base-cased model](https://huggingface.co/xlnet-base-cased) |
[📄 Paper for Dataset](https://aclanthology.org/2022.findings-naacl.60.pdf) | 
</div>

LLM_NER_MultiNERD is a Using the MultiNERD Named Entity Recognition (NER) dataset, complete the following instructions to train and 
evaluate a Named Entity Recognition model for English using BERT and XLNET.
Built on top of the familiar [🤗 Transformers](https://github.com/huggingface/transformers) library.

## Instructions:
### System A 
Fine-tune chosen model bert-base-cased and xlnet-base-cased on the English subset of the training set. 

### System B 
Train a model that will predict only five entity types and the `O` tag (I.e. not part of an entity). Therefore, you should perform the necessary pre-processing steps on the dataset. All examples should thus remain, but entity types not belonging to one of the following five should be set to zero: `PERSON(PER)`, `ORGANIZATION(ORG)`, `LOCATION(LOC)`, `DISEASES(DIS)`, Fine-tune chosen model on the filtered dataset.


## Setting up Docker environment 
Go to folder ```docker/```.
```
docker build -f Dockerfile -t NER-MultiNERD \
--build-arg username=$(username) .
```
```
docker run -it --shm-size 60G --gpus all \
-v /path/to/dir/:/home/username/NER-MultiNERD/ \
-v /path/to/storage/:/storage/ NER-MultiNERD
```
