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
    parser = argparse.ArgumentParser(description="Train a kana to kanji model using Transformers")
    
    # Training type
    parser.add_argument("--multi_gpu", action="store_true", default=False,
                       help="Enable multi-GPU training using Accelerator")

    # Dataset and model arguments
    parser.add_argument("--dataset_dir", required=True, help="Directory with the formatted dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to save the model")
    parser.add_argument("--model_name", default="google/mt5-small", 
                        help="Base model to fine-tune. Options include: google/mt5-small, google/mt5-base"
                        "google/mt5-large, google/mt5-xl, google/mt5-xxl")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Add training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Per device batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    
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
    run_name = os.path.basename(args.output_dir)
    if args.multi_gpu:
        if accelerator.is_main_process:
            wandb.init(project="kana-to-kanji", name=run_name)
    else:
        wandb.init(project="kana-to-kanji", name=run_name)

    # Load dataset
    dataset = load_from_disk(args.dataset_dir)
    print("Dataset structure:", dataset)
    print("First example:", dataset["train"][0] if "train" in dataset else dataset[0])
   
    # Load pretrained tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Define the preprocessing function
    def preprocess_function(examples):
        # Create inputs and targets
        inputs = ["translate kana to kanji: " + item["source"] for item in examples["translation"]]
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
   
    # Load CER metric only
    cer_metric = evaluate.load("cer")
   
    # Define compute metrics function for CER only
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
   
    # Initialize model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Add this code to make all parameters contiguous before DeepSpeed initialization
    for param in model.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        auto_find_batch_size=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=3,
        warmup_steps=500,
        bf16=True,
        group_by_length=True,
        report_to=["wandb", "tensorboard"],
        metric_for_best_model="cer",
        predict_with_generate=True,
        greater_is_better=False,
        load_best_model_at_end=True,
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
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
        eval_dataset=tokenized_dataset["test"],
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