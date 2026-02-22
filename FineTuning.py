import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel


def train(model, tokenizer, train_file, output_dir):
    dataset = load_dataset("json", data_files=train_file)

    def tokenize_function(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        full_text = prompt + example["response"]

        tokenized = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
        )

        labels = tokenized["input_ids"].copy()
        prompt_len = len(prompt_tokens["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        tokenized["labels"] = labels
        return tokenized

    tokenized_dataset = dataset["train"].map(tokenize_function)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    print("LoRA adapters saved successfully.")


def inference_loop(base_model,model, tokenizer):
    base_model.eval()
    model.eval()
    print("\n--- Entering Test Inference Mode ---")

    while True:
        instruction = input("\nEnter instruction (or type 'exit'): ")
        if instruction.lower() == "exit":
            break

        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output_base = base_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            output_model = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        response_base = tokenizer.decode(output_base[0], skip_special_tokens=True)
        response_model = tokenizer.decode(output_model[0], skip_special_tokens=True)
        print("\nModel Output before and after fine tuning:\n")
        print(response_base+"\n----------Base Respond Ends, FT Response starts\n")
        print(response_model)
        


def main():
    parser = argparse.ArgumentParser("Help Description")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training json file")
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()
    #if True: exit()

    print(f"Using model: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Apply LoRA ----
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ---- Train ----
    #train(model, tokenizer, args.train_file, args.output_dir)

    # ---- Reload Adapter Cleanly for Inference ----
    print("\nReloading model with trained LoRA adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model = PeftModel.from_pretrained(base_model, args.output_dir)

    # ---- Test Inference ----
    inference_loop(base_model, model, tokenizer)


if __name__ == "__main__":
    main()
