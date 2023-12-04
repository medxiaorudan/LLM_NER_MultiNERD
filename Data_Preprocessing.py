from datasets import load_dataset, DatasetDict

def data_preprocessing_A(original_dataset):
    data = original_dataset.filter(lambda example: example['lang'] == 'en')
    data = data.remove_columns(["lang"])

    ds = DatasetDict({
        'train': data['train'], 
        'eval': data['validation'], 
        'test': data['test']})

    label2id = {
        "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
        "B-LOC": 5, "I-LOC": 6, "B-ANIM": 7, "I-ANIM": 8, "B-BIO": 9,
        "I-BIO": 10, "B-CEL": 11, "I-CEL": 12, "B-DIS": 13, "I-DIS": 14, 
        "B-EVE": 15, "I-EVE": 16, "B-FOOD": 17, "I-FOOD": 18, "B-INST": 19, 
        "I-INST": 20, "B-MEDIA": 21, "I-MEDIA": 22, "B-MYTH": 23, "I-MYTH": 24, 
        "B-PLANT": 25, "I-PLANT": 26, "B-TIME": 27, "I-TIME": 28, "B-VEHI": 29, 
        "I-VEHI": 30
    }

    id2label = {tag: idx for idx, tag in label2id.items()}

    return ds, id2label, label2id

def data_preprocessing_B(original_dataset):
    data_filterd = original_dataset.filter(lambda example: example['lang'] == 'en')
    label_map = {
    "O": 0, "B-PER": 1,"I-PER": 2,"B-ORG": 3,"I-ORG": 4,
    "B-LOC": 5,"I-LOC": 6,"B-ANIM": 7,"I-ANIM": 8,"B-DIS": 13,
    "I-DIS": 14}

    valid_entity_map = {v: k for k, v in label_map.items()}

    def filter_and_transform(example):
        filtered_ner_tags = []
        for tag in example['ner_tags']:
            if tag in list(valid_entity_map.keys()) and tag ==13:
                filtered_ner_tags.append(9)  
            elif tag in list(valid_entity_map.keys()) and tag ==14:
                filtered_ner_tags.append(10)  
            elif tag in list(valid_entity_map.keys()):
                filtered_ner_tags.append(tag) 
            else:
                filtered_ner_tags.append(0) 
        
        return {
            'tokens': example['tokens'],
            'ner_tags': filtered_ner_tags,
            'lang': example['lang']
        }

    for split in ["train", "validation", "test"]:
        data_filterd[split] = data_filterd[split].map(filter_and_transform)

    data = data_filterd.remove_columns(["lang"])

    ds = DatasetDict({
    'trprocessed_dataset_Bain': data['train'], 
    'eval': data['validation'], 
    'test': data['test']})

    label2id = {
    "O": 0, "B-PER": 1,"I-PER": 2,"B-ORG": 3,"I-ORG": 4,
    "B-LOC": 5,"I-LOC": 6,"B-ANIM": 7,"I-ANIM": 8,"B-DIS": 9,
    "I-DIS": 10}

    id2label = {tag: idx for idx, tag in label2id.items()}

    return ds, id2label, label2id
