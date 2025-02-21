import argparse
from single_engine_subtopic import SubtopicEngine

def parse_args():
    parser = argparse.ArgumentParser(description='Run stance detection on subtopics and aggregate by speaker')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2', help='Model name')
    parser.add_argument('--wiki_model', type=str, default='sentence-transformers/all-mpnet-base-v2', help='Wiki model name')
    
    # System parameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU device number')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='ckp/best_model.pt', help='Path to load the trained model checkpoint')
    
    args = parser.parse_args()
    # Set wiki_model freezing to match main model freezing
    args.n_layers_freeze = 4  # Default value from original script
    args.n_layers_freeze_wiki = args.n_layers_freeze
    return args

def main():
    args = parse_args()
    print("Starting with configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    engine = SubtopicEngine(args)
    engine.predict()

if __name__ == '__main__':
    main()
