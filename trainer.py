from transformers import Trainer, TrainingArguments
import torch
from torch import nn


import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTrainerA(Trainer):
    def compute_loss(self, 
                     model, 
                     inputs, 
                     return_outputs=False):
        
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss 
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
             9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
             16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
             23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0], 
            device=get_device())
        )
        loss = loss_fct(logits.view(-1, 
                                    self.model.config.num_labels), 
                        labels.view(-1)
                        )
        return (loss, outputs) if return_outputs else loss


def TrainerA(args, model, data_collator, compute_metrics, tokenizer, train_dataset, eval_dataset):
    trainer = CustomTrainerA(model=model,
                           args=args,
                           data_collator=data_collator,
                           compute_metrics=compute_metrics,
                           tokenizer=tokenizer,
                           train_dataset=train_dataset,
                           eval_dataset=eval_dataset)
    return trainer

class CustomTrainerB(Trainer):
    def compute_loss(self, 
                     model, 
                     inputs, 
                     return_outputs=False):
        
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss 
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
             9.0, 10.0, 11.0], 
            device=get_device())
        )
        loss = loss_fct(logits.view(-1, 
                                    self.model.config.num_labels), 
                        labels.view(-1)
                        )
        return (loss, outputs) if return_outputs else loss

def TrainerB(args, model, data_collator, compute_metrics, tokenizer, train_dataset, eval_dataset):
    trainer = CustomTrainerA(model=model,
                           args=args,
                           data_collator=data_collator,
                           compute_metrics=compute_metrics,
                           tokenizer=tokenizer,
                           train_dataset=train_dataset,
                           eval_dataset=eval_dataset)
    return trainer