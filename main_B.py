import argparse
import sys
sys.path.append("./")

from Data_Preprocessing import data_preprocessing_B
from utils import tokenize_and_align_labels, compute_metrics
from trainer import CustomTrainerB
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset
import torch 
import evaluate

def main_B(args, train_args):

    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DEVICE = get_device()

    original_dataset = load_dataset('Babelscape/multinerd')

    ds, id2label, label2id = data_preprocessing_B(original_dataset)

    pos_tag_values = list(label2id.keys())
    NUM_OF_LABELS = len(pos_tag_values)

    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_CKPT)
    def tokenize_and_align_labels(samples):
        tokenized_inputs = tokenizer(samples["tokens"], 
                                        truncation=True, 
                                        is_split_into_words=True)
        
        labels = []
        for idx, label in enumerate(samples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            prev_word_idx = None
            label_ids = []
            for word_idx in word_ids: # set special tokens to -100
                if word_idx is None or word_idx == prev_word_idx:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
                prev_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    encoded_ds = ds.map(tokenize_and_align_labels, 
                        batched=True, 
                        remove_columns=
                            [
                                'ner_tags', 
                                'tokens'
                            ]
                        )
    model = (AutoModelForTokenClassification.from_pretrained(
    args.MODEL_CKPT,
    num_labels=NUM_OF_LABELS,
    id2label=id2label,
    label2id=label2id
    ).to(DEVICE))

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = CustomTrainerB(model, 
                    args=train_args,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    tokenizer=tokenizer,
                    train_dataset=encoded_ds["train"],
                    eval_dataset=encoded_ds["eval"],
                    )

    train_results = trainer.train()

    trainer.save_model(OUTPUT_DIR)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    model = trainer.model
    model.eval()    # switch to evaluation mode

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument("--MODEL_CKPT", default="bert-base-cased", required=True, type=str, help="The pre-trained model load from hugging face")
    
    MODEL_CKPT = args.MODEL_CKPT
    MODEL_NAME = f"{args.MODEL_CKPT}-finetuned-MultiNERD-SystemA"
    NUM_OF_EPOCHS = 2
    BATCH_SIZE = 12
    STRATEGY = "epoch"
    REPORTS_TO = "tensorboard"
    WEIGHT_DECAY = 0.01
    LR = 2e-5
    STEPS = 1250
    OUTPUT_DIR = f'/srv/users/rudxia/Developer_NLP/notebooks/results/Outdir/{args.MODEL_CKPT}-finetuned-MultiNERD-SystemA'
    LOG_DIR= f'/srv/users/rudxia/Developer_NLP/notebooks/results/Log/{args.MODEL_CKPT}-finetuned-MultiNERD-SystemA'

    train_args = TrainingArguments(
        OUTPUT_DIR,    
        MODEL_NAME,
        log_level="error",
        logging_first_step=True,
        logging_dir = LOG_DIR,
        learning_rate=LR,
        num_train_epochs=NUM_OF_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy=STRATEGY,
        report_to=REPORTS_TO,
        disable_tqdm=False,
        logging_steps=STEPS,
        weight_decay=WEIGHT_DECAY,
        save_strategy=STRATEGY,
        hub_private_repo=False,
        push_to_hub=False
    )
  
    main_B(args,train_args)
