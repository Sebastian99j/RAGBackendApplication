from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import uuid

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def fine_tune_gpt2(req):
    texts = [f"User: {ex.input}\nAI: {ex.output}" for ex in req.training_data]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./gpt2-temp-output",
        overwrite_output_dir=True,
        num_train_epochs=req.epochs,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_total_limit=1,
        save_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    save_path = f"./gpt2-finetuned-{uuid.uuid4().hex[:8]}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return {
        "status": "success",
        "message": f"Model został przetrenowany i zapisany w {save_path}"
    }
