# train.py
import argparse
from engine import Engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="zero-shot-stance/data/VAST/vast_train.csv", help="Path to training CSV")
    parser.add_argument("--dev_file", type=str, default="zero-shot-stance/data/VAST/vast_dev.csv", help="Path to development CSV")
    parser.add_argument("--test_file", type=str, default="zero-shot-stance/data/VAST/vast_test.csv", help="Path to test CSV")
    parser.add_argument("--model_path", type=str, default="/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/models", help="Path to nv-embed-v2 model directory")
    parser.add_argument("--save_path", type=str, default="best_model.pt", help="Path to save the best model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay")
    parser.add_argument("--num_labels", type=int, default=3, help="Number of stance labels")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for classifier")

    # PEFT/LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    
    # Instruction strings (you can adjust these as needed).
    parser.add_argument("--doc_instruction", type=str, default="Instruct: Encode the following document and topic jointly for stance detection on the topic.", help="Instruction for document encoding")
    parser.add_argument("--wiki_instruction", type=str, default="Instruct: Encode the following Wikipedia article to provide context for stance detection on this topic.", help="Instruction for Wikipedia encoding")
    parser.add_argument("--doc_prefix", type=str, default="Document and topic: ", help="Prefix for document and topic text")
    parser.add_argument("--wiki_prefix", type=str, default="Wikipedia article: ", help="Prefix for wiki text")

    args = parser.parse_args()

    engine = Engine(args)
    engine.train()

if __name__ == "__main__":
    main()
