import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./sia_distilBERT_mismatch_model"
DATA_PATH = "your_dataset.csv" 

def load_and_prepare_data(data_path):
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print("Data source file path absent. Synthesizing mockup dataframe verification loop...")
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'text': np.random.choice(["Outage on database 500 server error", "Typo on front page layout"], n_samples),
            'human_priority': np.random.choice(['Low', 'Critical'], n_samples),
            'customer_domain': np.random.choice(['Enterprise', 'Free Tier'], n_samples),
            'resolution_time_hrs': np.random.randint(1, 100, n_samples)
        })
        df['label'] = df.apply(lambda r: 1 if "500" in r['text'] and r['human_priority'] == 'Low' else 0, axis=1)
    
    df['input_text'] = df.apply(
        lambda r: f"Domain: {r['customer_domain']} | Priority: {r['human_priority']} | Resolution Time: {r['resolution_time_hrs']}h | Ticket: {r['text']}", 
        axis=1
    )
    return df[['input_text', 'label']]

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights) if self.class_weights is not None else torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    df = load_and_prepare_data(DATA_PATH)
    labels = df['label'].values
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=labels)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_fn(x): return tokenizer(x['input_text'], truncation=True, padding='max_length', max_length=128)
    
    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, evaluation_strategy="epoch", save_strategy="epoch",
        learning_rate=2e-5, per_device_train_batch_size=32, num_train_epochs=1, report_to="none"
    )
    
    trainer = CustomTrainer(class_weights=weights, model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()