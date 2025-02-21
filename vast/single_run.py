import argparse
from single_engine import Engine

def parse_args():
    parser = argparse.ArgumentParser(description='Run single training/inference with VAST model')
    
    # Model and data parameters
    parser.add_argument('--data', type=str, default='vast', help='Dataset name')
    parser.add_argument('--topic', type=str, default='', help='Topic to use')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2', help='Model name')
    parser.add_argument('--wiki_model', type=str, default='sentence-transformers/all-mpnet-base-v2', help='Wiki model name')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=8e-6, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_layers_freeze', type=int, default=4, help='Number of layers to freeze')
    parser.add_argument('--l2_reg', type=float, default=4e-5, help='L2 regularization weight')
    
    # System parameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU device number')
    parser.add_argument('--inference', type=int, default=1, help='0 for training, 1 for inference')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='ckp/best_model.pt', help='Path to save the best model checkpoint')
    
    args = parser.parse_args()
    # Set wiki_model freezing to match main model freezing
    args.n_layers_freeze_wiki = args.n_layers_freeze
    return args

def main():
    args = parse_args()
    print("Starting with configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    engine = Engine(args)
    engine.train()

if __name__ == '__main__':
    main()
