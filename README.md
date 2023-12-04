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
