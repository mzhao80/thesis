import argparse
import os
import pandas as pd
from split_long_documents import main as split_documents
from split_engine_subtopic import SplitSubtopicEngine

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with document splitting')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help='Model name for tokenizer')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--checkpoint_path', type=str, default="ckp_new/finetuned_model.pt",
                        help='Path to the model checkpoint file (default: ckp_new/finetuned_model.pt)')
    
    # Document splitting arguments
    parser.add_argument('--process_original', action='store_true', default=False,
                        help='Process the original taxonomy data by splitting documents (default: True)')
    parser.add_argument('--max_length', type=int, default=384,
                        help='Maximum token length per document chunk (default: 384 to leave room for target and summary)')
    parser.add_argument('--overlap', type=int, default=50,
                        help='Number of tokens to overlap between chunks')
    
    # Aggregation method arguments
    parser.add_argument('--chunk_aggregation_method', type=str, default='max_pooling',
                        choices=['max_pooling', 'simple_averaging', 'weighted_averaging'],
                        help='Method to aggregate document chunks (default: max_pooling)')
    parser.add_argument('--document_aggregation_method', type=str, default='max_pooling',
                        choices=['max_pooling', 'simple_averaging', 'weighted_averaging'],
                        help='Method to aggregate documents into subtopic scores (default: max_pooling)')
    
    # Input/output arguments
    parser.add_argument('--input_csv', type=str,
                        default='/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data.csv',
                        help='Path to input CSV file (original data if using --process_original, otherwise pre-split data)')
    parser.add_argument('--split_csv', type=str,
                        default='/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data_split.csv',
                        help='Path to save the split documents (if --process_original is used)')
    parser.add_argument('--output_csv', type=str,
                        default='/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/stance_predictions.csv',
                        help='Path to save the stance prediction results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Process original data if requested
    if args.process_original:
        print(f"Processing original taxonomy data from {args.input_csv}")
        print(f"Using max_length={args.max_length}, overlap={args.overlap}")
        
        # Set up arguments for split_documents
        import sys
        sys.argv = [
            'split_long_documents.py',
            f'--max_length={args.max_length}',
            f'--overlap={args.overlap}',
            f'--input_csv={args.input_csv}',
            f'--output_csv={args.split_csv}'
        ]
        
        try:
            # Run the document splitting
            print("\n===== Splitting documents =====")
            split_documents()
            print("===== Document splitting complete =====\n")
            
            # Update input CSV to the split data
            input_csv = args.split_csv
        except Exception as e:
            print(f"Error during document splitting: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # Use provided input CSV
        input_csv = args.input_csv
    
    print(f"Running inference on {input_csv}")
    
    # Initialize the engine and run inference
    try:
        # Initialize the engine
        print("\n===== Initializing model =====")
        engine = SplitSubtopicEngine(
            model_name=args.model_name,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint_path,
            chunk_aggregation_method=args.chunk_aggregation_method,
            document_aggregation_method=args.document_aggregation_method
        )
        
        # Run inference on the split data
        print("\n===== Running inference =====")
        print(f"Using chunk aggregation method: {args.chunk_aggregation_method}")
        print(f"Using document aggregation method: {args.document_aggregation_method}")
        results = engine.predict(input_csv)
        
        # Save results
        if results is not None:
            print(f"\nSaving results to {args.output_csv}")
            results.to_csv(args.output_csv, index=False)
            print(f"Saved {len(results)} predictions")
            return True
        else:
            print("No results to save")
            return False
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n===== Processing complete =====")
        else:
            print("\n===== Processing completed with errors =====")
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
