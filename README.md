<div align="center">
<h1>
LLM_NER_MultiNERD
</h1>
<center>
![image](https://github.com/medxiaorudan/LLM_NER_MultiNERD/assets/22127304/6bcafce0-e2d1-4074-8588-828ea6f1ba48)
</center>

[🤗 MultiNERD Dataset](https://huggingface.co/models?library=span-marker) |
[🛠️ bert-base-cased model](https://huggingface.co/bert-base-cased) |
[🛠️ xlnet-base-cased model](https://huggingface.co/xlnet-base-cased) |
[📄 Paper](https://aclanthology.org/2022.findings-naacl.60.pdf) | 
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
