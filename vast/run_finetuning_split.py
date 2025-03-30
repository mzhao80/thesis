import argparse
import os
import torch
from finetuning_split_engine import SplitFinetuningEngine
from finetuning_split_datasets import create_train_val_test_split


def main():
    parser = argparse.ArgumentParser(description='Fine-tune stance detection model on split documents')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='/n/holylabs/LABS/arielpro_lab/Lab/michaelzhao/taxonomy_data_2023_sample_stance.csv',
                       help='Path to the CSV file with training data')
    parser.add_argument('--output_dir', type=str, default='split_finetuning_data',
                       help='Directory to save processed data splits')
    parser.add_argument('--train_path', type=str, default=None,
                       help='Path to training data CSV (if already split)')
    parser.add_argument('--val_path', type=str, default=None,
                       help='Path to validation data CSV (if already split)')
    parser.add_argument('--test_path', type=str, default=None,
                       help='Path to test data CSV (if already split)')
    parser.add_argument('--process_original', action='store_true',
                       help='Process original documents (split them)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of data to use for validation')
    parser.add_argument('--split_seed', type=int, default=42,
                       help='Random seed for data splitting')
    # Model parameters
    parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                       help='Pretrained model name')
    parser.add_argument('--ckpt_path', type=str, default='ckp/best_model.pt',
                       help='Path to load checkpoint from')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--max_doc_length', type=int, default=384,
                       help='Maximum document length for splitting')
    parser.add_argument('--overlap', type=int, default=50,
                       help='Overlap between document chunks')
    parser.add_argument('--n_layers_freeze', type=int, default=4,
                       help='Number of layers to freeze in the model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--l2_reg', type=float, default=0.01,
                       help='L2 regularization weight')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=5,
                       help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_path', type=str, default='ckp/finetuned_model.pt',
                       help='Path to save the model')
    
    # Other parameters
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU index')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--chunk_aggregation_method', type=str, default='max_pooling',
                       choices=['max_pooling', 'simple_averaging', 'weighted_averaging'],
                       help='Method to aggregate document chunks (max_pooling, simple_averaging, weighted_averaging)')
    
    args = parser.parse_args()
    
    # Create data splits if paths are not provided
    if args.train_path is None or args.val_path is None or args.test_path is None:
        print(f"Creating train/val/test splits from {args.data_path}")
        
        # Debug the source data
        print("Analyzing source data...")
        import pandas as pd
        source_df = pd.read_csv(args.data_path, index_col=False)
        source_df.dropna(subset=['target', 'label'], inplace=True)
        source_df.to_csv(args.data_path, index=False)
        print(f"Source data shape: {source_df.shape}")
        print(f"Columns: {source_df.columns.tolist()}")
        
        train_path, val_path, test_path = create_train_val_test_split(
            csv_path=args.data_path,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            process_original=args.process_original,
            max_doc_length=args.max_doc_length,
            overlap=args.overlap,
            model=args.model,
            seed=args.split_seed
        )
        
        args.train_path = train_path
        args.val_path = val_path
        args.test_path = test_path
    
    # Create engine and train
    engine = SplitFinetuningEngine(args, chunk_aggregation_method=args.chunk_aggregation_method)
    best_f1 = engine.train()
    
    print(f"Training complete! Best validation F1: {best_f1:.4f}")
    print(f"Model saved to: {args.save_path}")
    
    # Run inference on test set
    print("Running inference on test set...")
    print(f"Using chunk aggregation method: {args.chunk_aggregation_method}")
    test_predictions, test_aggregated = engine.predict(args.test_path, chunk_aggregation_method=args.chunk_aggregation_method)
    
    # Save predictions
    test_predictions_path = os.path.join(args.output_dir, 'test_predictions.csv')
    test_aggregated_path = os.path.join(args.output_dir, 'test_aggregated.csv')
    
    test_predictions.to_csv(test_predictions_path, index=False)
    test_aggregated.to_csv(test_aggregated_path, index=False)
    
    print(f"Test predictions saved to: {test_predictions_path}")
    print(f"Aggregated test predictions saved to: {test_aggregated_path}")


if __name__ == '__main__':
    main()
