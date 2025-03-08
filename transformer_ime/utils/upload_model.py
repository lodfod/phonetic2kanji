import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload a trained model to Hugging Face Hub")
    parser.add_argument("--model_path", required=True, help="Path to the trained model directory")
    parser.add_argument("--hf_repo_id", required=True, help="Hugging Face repository ID (e.g., 'username/model-name')")
    parser.add_argument("--commit_message", default="Upload trained model", help="Commit message for the upload")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    parser.add_argument("--readme", default=None, help="Path to a README.md file to include with the model")
    args = parser.parse_args()

    print(f"Loading model and tokenizer from {args.model_path}")
    
    # Load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Create repository (if it doesn't exist)
    api = HfApi()
    
    try:
        api.create_repo(
            repo_id=args.hf_repo_id,
            private=args.private,
            exist_ok=True
        )
        print(f"Repository {args.hf_repo_id} created or already exists")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Upload README if provided
    if args.readme:
        try:
            api.upload_file(
                path_or_fileobj=args.readme,
                path_in_repo="README.md",
                repo_id=args.hf_repo_id,
                commit_message="Upload README"
            )
            print(f"README uploaded to {args.hf_repo_id}")
        except Exception as e:
            print(f"Error uploading README: {e}")
    
    # Push the model and tokenizer to the Hub
    print(f"Uploading model and tokenizer to {args.hf_repo_id}")
    model.push_to_hub(args.hf_repo_id, commit_message=args.commit_message)
    tokenizer.push_to_hub(args.hf_repo_id, commit_message=args.commit_message)
    
    print(f"Model and tokenizer successfully uploaded to {args.hf_repo_id}")
    print(f"View your model at: https://huggingface.co/{args.hf_repo_id}")

if __name__ == "__main__":
    main()