import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

def load_and_prepare_data(csv_path):
    """Load and prepare the training data"""
    print("Loading training data...")
    
    # Load CSV without header since you deleted the header row
    df = pd.read_csv(csv_path, header=None, names=['question', 'analogy', 'explanation', 'difficulty', 'domain'])
    
    # Debug: Print info
    print(f"Loaded {len(df)} rows")
    print(f"First row: {df.iloc[0].to_dict()}")
    
    # Create training prompts
    training_texts = []
    for _, row in df.iterrows():
        try:
            prompt = f"""<|system|>You are AnalogyGPT, an expert at creating simple, clever analogies to explain complex concepts.<|end|>
<|user|>{row['question']}<|end|>
<|assistant|>ANALOGY: {row['analogy']}

EXPLANATION: {row['explanation']}<|end|>"""
            training_texts.append(prompt)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue
    
    print(f"Created {len(training_texts)} training examples")
    return training_texts

def setup_model_and_tokenizer():
    """Initialize the model and tokenizer"""
    print("Loading Phi-3-mini model...")
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager"  # Avoid flash attention warnings
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Prepare model for training
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora_config():
    """Configure LoRA for efficient fine-tuning"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Low rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    return lora_config

def create_dataset(texts, tokenizer, max_length=512):
    """Create tokenized dataset"""
    print("Tokenizing data...")
    
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        # Add labels for language modeling (labels = input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Create dataset from texts
    dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"]  # Remove the original text column
    )
    
    print(f"Tokenized dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

def train_model(csv_path="training_data.csv"):
    """Main training function"""
    print("Starting AnalogyGPT training...")
    
    # Load data
    training_texts = load_and_prepare_data(csv_path)
    print(f"Loaded {len(training_texts)} training examples")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Apply LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create dataset
    train_dataset = create_dataset(training_texts, tokenizer)
    
    # Training arguments optimized for your hardware
    training_args = TrainingArguments(
        output_dir="./analogygpt-phi3-mini",
        num_train_epochs=2,  # Reduced from 3 to avoid issues
        per_device_train_batch_size=1,  # Small batch for 4GB VRAM
        gradient_accumulation_steps=4,  # Effective batch size = 4
        warmup_steps=50,  # Reduced warmup
        learning_rate=5e-5,  # Slightly lower learning rate
        fp16=True,  # Memory efficient
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=False,
        gradient_checkpointing=False,  # Disabled to avoid conflicts
        max_grad_norm=1.0,  # Add gradient clipping
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked LM
        pad_to_multiple_of=8,  # Efficiency optimization
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained("./analogygpt-phi3-mini")
    
    print("Training completed! Model saved to ./analogygpt-phi3-mini")

if __name__ == "__main__":
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Start training
    train_model()