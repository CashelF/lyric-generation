from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

dataset = load_dataset("amishshah/song_lyrics")
train_test_dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = train_test_dataset["train"]
val_dataset = train_test_dataset["test"]

# Prepare the dataset (ensure your data is in a format the model expects)
# Here you might need a custom dataset class if using more complex inputs
train_encodings = tokenizer(train_dataset, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_dataset, truncation=True, padding=True, max_length=512)

# Define training args
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=4,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=val_encodings
)

# Train the model
trainer.train()

trainer.evaluate()

# Eval
outputs = model(**inputs)
embeddings = outputs.last_hidden_state

from sklearn.metrics.pairwise import cosine_similarity

# Suppose `emb1` and `emb2` are the embeddings of two different texts
similarity = cosine_similarity(emb1.mean(dim=0, keepdim=True), emb2.mean(dim=0, keepdim=True))
print(f"Cosine similarity: {similarity.item()}")