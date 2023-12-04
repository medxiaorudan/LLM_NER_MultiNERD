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

    encoded_ds = ds.map(lambda x: tokenize_and_align_labels(tokenizer, x), 
                    batched=True, 
                    remove_columns=['ner_tags', 'tokens'])

    model = (AutoModelForTokenClassification.from_pretrained(
    args.MODEL_CKPT,
    num_labels=NUM_OF_LABELS,
    id2label=id2label,
    label2id=label2id
    ).to(DEVICE))

    label_list = pos_tag_values

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = CustomTrainerB(model, 
                    args=train_args,
                    data_collator=data_collator,
                    compute_metrics=lambda p: compute_metrics(p, label_list),  
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--MODEL_CKPT", default="bert-base-cased", required=True, type=str, help="The pre-trained model load from hugging face")
    args = parser.parse_args()

    MODEL_CKPT = args.MODEL_CKPT
    MODEL_NAME = f"{args.MODEL_CKPT}-finetuned-MultiNERD-SystemB"
    NUM_OF_EPOCHS = 2
    BATCH_SIZE = 12
    STRATEGY = "epoch"
    REPORTS_TO = "tensorboard"
    WEIGHT_DECAY = 0.01
    LR = 2e-5
    STEPS = 1250
    OUTPUT_DIR = f'/srv/users/rudxia/Developer_NLP/notebooks/results/Outdir/{args.MODEL_CKPT}-finetuned-MultiNERD-SystemB'
    LOG_DIR= f'/srv/users/rudxia/Developer_NLP/notebooks/results/Log/{args.MODEL_CKPT}-finetuned-MultiNERD-SystemB'

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
  
    main_B(args, train_args)
