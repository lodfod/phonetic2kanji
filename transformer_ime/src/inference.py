import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Convert kana to kanji using a fine-tuned model")
    parser.add_argument("--model_dir", required=True, help="Directory with the fine-tuned model")
    parser.add_argument("--input", required=True, help="Kana input to convert")
    parser.add_argument("--tokenizer_type", default="mecab", choices=["mecab", "character"], 
                        help="Type of Japanese tokenizer to use")
    parser.add_argument("--batch", action="store_true", help="Process input as a batch file")
    # Add generation parameters
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of generated text")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_dir}...")
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    
    # Load a Japanese-specific tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # Test the tokenizer with Japanese text
    test_text = "コトハ"
    test_tokens = tokenizer.tokenize(test_text)
    print(f"Test tokenization of '{test_text}': {test_tokens}")
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using {device} device")
    
    model = model.to(device)
    
    # Set generation parameters
    generation_params = {
        "max_length": 128,
        "do_sample": False,
        "top_k": 50,
        "top_p": 1.0,
        "temperature": 1.0,
        "num_beams": 1,
        "min_length": 1,
        "no_repeat_ngram_size": 2,
    }
    print(f"Generation parameters: {generation_params}")
    
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
            
            outputs = model.generate(input_ids, **generation_params)
            
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
        
        outputs = model.generate(input_ids, **generation_params)
        
        # Decode the output
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Print the results
        print(f"Kana: {args.input}")
        print(f"Kanji: {decoded_output}")

def load_model(model_dir):
    # Load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = get_japanese_tokenizer()
    
    # Ensure the tokenizer can handle Japanese characters
    # If your tokenizer was not trained with Japanese characters, you might need to use a different one
    # or add special handling for Japanese input
    
    return model, tokenizer

def generate_text(model, tokenizer, input_text, device, generation_params=None):
    # Format the input properly for the task
    # For Japanese kana to kanji conversion, you might want to use a specific format
    formatted_input = f"translate kana to kanji: {input_text}"
    
    # Tokenize the input
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
    
    if generation_params is None:
        generation_params = {
            "max_length": 128,
            "do_sample": False,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 1.0,
            "num_beams": 1,
            "min_length": 1,
            "no_repeat_ngram_size": 2,
        }
    
    # Generate the output
    outputs = model.generate(**inputs, **generation_params)
    
    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return decoded_output

if __name__ == "__main__":
    main() 