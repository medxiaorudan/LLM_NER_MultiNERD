<div align="center">
<h1>
LLM_NER_MultiNERD
</h1>
<center>
![image](https://github.com/medxiaorudan/LLM_NER_MultiNERD/assets/22127304/33c16521-5b7b-44c3-b92c-170b6aabee2c)
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
