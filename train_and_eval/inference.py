import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Convert kana to kanji using a fine-tuned model")
    parser.add_argument("--model_name", required=True, help="Model name or path")
    parser.add_argument("--input", required=True, help="Kana input to convert")
    parser.add_argument("--batch", action="store_true", help="Process input as a batch file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_name}...")
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using {device} device")
    
    model = model.to(device)
    
    print(f"Using model's default generation parameters with max_length={args.max_length}")
    
    if args.batch:
        # Process as a batch file
        with open(args.input, 'r', encoding='utf-8') as f:
            kana_lines = [line.strip() for line in f if line.strip()]
        
        kanji_lines = []
        for kana in kana_lines:
            input_text = f"translate kana to kanji: {kana}"
            if args.debug:
                print(f"Input text: {input_text}")
            
            # Only pass input_ids to generate
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            if args.debug:
                print(f"Input token IDs: {input_ids}")
                print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
            
            # Use default generation parameters with max_length
            outputs = model.generate(input_ids, max_length=args.max_length)
            
            # Decode the output
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Print the results
            print(f"Kana: {kana}")
            print(f"Kanji: {decoded_output}")
            
            kanji_lines.append(decoded_output)
    else:
        # Process single input
        input_text = f"translate kana to kanji: {args.input}"
        if args.debug:
            print(f"Input text: {input_text}")
        
        # Only pass input_ids to generate
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        if args.debug:
            print(f"Input token IDs: {input_ids}")
            print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
        
        # Use default generation parameters with max_length
        outputs = model.generate(input_ids, max_length=args.max_length)
        
        # Decode the output
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Print the results
        print(f"Kana: {args.input}")
        print(f"Kanji: {decoded_output}")

def generate_text(model, tokenizer, input_text, device, max_length=128):
    # Format the input properly for the task
    # For Japanese kana to kanji conversion, you might want to use a specific format
    formatted_input = f"translate kana to kanji: {input_text}"
    
    # Tokenize the input
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
    
    # Generate the output with default parameters and max_length
    outputs = model.generate(**inputs, max_length=max_length)
    
    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return decoded_output

if __name__ == "__main__":
    main() 