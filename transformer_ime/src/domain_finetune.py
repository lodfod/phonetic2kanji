import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np
from datasets import load_from_disk
import torch
import wandb
from accelerate import Accelerator

def main():
    parser = argparse.ArgumentParser(description="Domain-specific fine-tuning for kana to kanji model")
    
    # Training type
    parser.add_argument("--multi_gpu", action="store_true", default=False,
                       help="Enable multi-GPU training using Accelerator")

    # Dataset and model arguments
    parser.add_argument("--dataset_dir", required=True, 
                        help="Directory with the domain-specific dataset")
    parser.add_argument("--output_dir", required=True, 
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--model_name", required=True,
                        help="HuggingFace model ID or path to the pre-trained model")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Fine-tuning specific hyperparameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per device batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Portion of training to perform learning rate warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay to apply during fine-tuning")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--freeze_encoder", action="store_true", default=False,
                        help="Freeze encoder parameters and only fine-tune the decoder")
    parser.add_argument("--domain_prefix", type=str, default="",
                        help="Optional domain prefix to add to inputs (e.g., 'medical: ')")
    
    args = parser.parse_args()

    # Set up training environment
    if args.multi_gpu:
        setup_multi_GPU_training_environment()
        if not torch.cuda.is_available():
            raise ValueError("Multi-GPU training requested but CUDA is not available")
        accelerator = Accelerator()
        print("Multi-GPU training enabled")
    else:
        setup_single_GPU_training_environment()
        accelerator = None
        # Set up device for training
        if torch.cuda.is_available():
            print("Using CUDA device for single GPU accelerated training")
        elif torch.backends.mps.is_available():
            print("Using MPS device for accelerated training")
        else:
            print("No accelerated device found, using CPU")

    # Initialize wandb with run name from output_dir
    run_name = f"domain-finetune-{os.path.basename(args.output_dir)}"
    if args.multi_gpu:
        if accelerator.is_main_process:
            wandb.init(project="kana-to-kanji-domain-finetune", name=run_name)
    else:
        wandb.init(project="kana-to-kanji-domain-finetune", name=run_name)

    # Load domain-specific dataset
    dataset = load_from_disk(args.dataset_dir)
    print("Dataset structure:", dataset)
    print("First example:", dataset["train"][0] if "train" in dataset else dataset[0])
   
    # Load pretrained tokenizer from the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Define the preprocessing function with optional domain prefix
    def preprocess_function(examples):
        # Create inputs and targets with optional domain prefix
        prefix = f"{args.domain_prefix} " if args.domain_prefix else ""
        inputs = [f"translate kana to kanji: {prefix}{item['source']}" for item in examples["translation"]]
        targets = [item["target"] for item in examples["translation"]]

        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True)
       
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_length, truncation=True)
           
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
   
    # Preprocess the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
   
    # Load CER metric
    cer_metric = evaluate.load("cer")
   
    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Handle sequence generation output format
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Replace any out-of-range values with pad token id to prevent overflow
        preds = np.where(
            (preds < 0) | (preds > tokenizer.vocab_size), 
            tokenizer.pad_token_id, 
            preds
        )
        
        # Decode predictions
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode labels
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Filter out empty strings
        filtered_pairs = [(p, r) for p, r in zip(preds, labels) if len(r.strip()) > 0]
        if not filtered_pairs:
            return {"cer": 1.0}  # Return worst score if all references are empty
        filtered_preds, filtered_labels = zip(*filtered_pairs)
        
        # Compute CER score
        cer_score = cer_metric.compute(predictions=filtered_preds, references=filtered_labels)
        
        return {"cer": cer_score}
   
    # Initialize model from pre-trained checkpoint
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Optionally freeze encoder parameters
    if args.freeze_encoder:
        print("Freezing encoder parameters")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Add this code to make all parameters contiguous before DeepSpeed initialization
    for param in model.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()

    # Calculate warmup steps based on warmup ratio
    total_steps = len(tokenized_dataset["train"]) // (args.batch_size * args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Define training arguments optimized for fine-tuning
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        auto_find_batch_size=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        warmup_steps=warmup_steps,
        bf16=True,
        group_by_length=True,
        report_to=["wandb", "tensorboard"],
        metric_for_best_model="cer",
        predict_with_generate=True,
        greater_is_better=False,
        load_best_model_at_end=True,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
        # Fine-tuning specific parameters
        fp16_full_eval=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        generation_max_length=args.max_length,
    )
   
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
   
    # Initialize trainer with updated parameters
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
   
    # Handle model preparation based on mode
    if args.multi_gpu:
        if accelerator.is_main_process:
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        model, trainer = accelerator.prepare(model, trainer)

    # Train model
    trainer.train()
   
    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
   
    # Save the tokenizer
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
   
    # Close wandb
    if args.multi_gpu:
        if accelerator.is_main_process:
            wandb.finish()
    else:
        wandb.finish()

# Setup training for single gpu
def setup_single_GPU_training_environment():
    # Disable distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU only
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    # Clear any existing distributed training settings
    for var in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        if var in os.environ:
            del os.environ[var]

# Setup training for multi gpu
def setup_multi_GPU_training_environment():
    os.environ["NCCL_TIMEOUT"] = "3600"
    os.environ["NCCL_IB_TIMEOUT"] = "120"
    os.environ["NCCL_SOCKET_TIMEOUT"] = "120"
    os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000"
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 

if __name__ == "__main__":
    main() 