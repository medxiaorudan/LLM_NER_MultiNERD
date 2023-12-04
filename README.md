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
Train a model that will predict only five entity types and the `O` tag (I.e. not part of an entity). Therefore, the necessary [pre-processing](https://github.com/medxiaorudan/LLM_NER_MultiNERD/blob/main/Data_Preprocessing.py) steps should be performed on the dataset. All examples should thus remain, but entity types not belonging to one of the following five should be set to zero: `PERSON(PER)`, `ORGANIZATION(ORG)`, `LOCATION(LOC)`, `DISEASES(DIS)`, `ANIMAL(ANIM)`. Fine-tune the chosen models on the filtered dataset.

### BERT
BERT (Bidirectional Encoder Representations from Transformers) employs a bidirectional attention mechanism to capture contextual information from both left and right contexts. It uses pre-training tasks, such as masked language modeling, to learn contextualized embeddings. 

### XLNet
XLNet improves upon BERT by introducing permutation language modeling. It captures bidirectional context like BERT but allows for a more flexible information flow. In Named Entity Recognition (NER) tasks, these models excel at understanding the relationships between words and recognizing entities such as persons, organizations, and locations. Their deep contextual embeddings enable them to capture nuanced patterns, improving accuracy in identifying named entities within text.

## Setting up the Docker environment and installing the dependencies
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
You can install the following dependencies to run tasks in the environment:
```bash
pip install -r requirements.txt
```
### Input Format

The BIOS tag scheme of the input, with each character its label for one line. Sentences are split with a null line.

<left>
<img src="https://github.com/medxiaorudan/LLM_NER_MultiNERD/blob/main/images/BIOS.PNG" width="220"> 
</left>

### Run the code

The fine-tune BERT for System A:
```bash
python main_A.py --MODEL_CKPT bert-base-cased
```
The fine-tune BERT for System B:
```bash
python main_B.py --MODEL_CKPT bert-base-cased
```
The fine-tune XLNET for System A:
```bash
python main_A.py --MODEL_CKPT xlnet-base-cased
```
The fine-tune XLNET for System B:
```bash
python main_B.py --MODEL_CKPT xlnet-base-cased
```
### CLUENER result

The overall performance of BERT and XLNET on **dev**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+SystemA | 0.7897     | 0.8031     | 0.7963    |
| BERT+SystemB     | 0.7977 | 0.8177 | 0.8076 |
| XLNET+SystemA    | 0.8132 | 0.8092 | 0.8112 |
| XLNET+SystemB    | 0.8267 | 0.8073 | **0.8169** |

