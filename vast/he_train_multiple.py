from he_engine_multiple import Engine

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=('vast'), default='vast',
                        help='which dataset to use')
    parser.add_argument('--topic', type=str, choices=(''), default='',
                        help='the topic to use')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--l2_reg', type=float, default=5e-5)
    parser.add_argument('--max_grad', type=float, default=0)
    parser.add_argument('--n_layers_freeze', type=int, default=10)
    parser.add_argument('--model', type=str,
                        choices=('bert-base-uncased', 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/all-mpnet-base-v2'),
                        default='bert-base-uncased',
                        help='choose between bert-base-uncased and all-MiniLM-L6-v2 and all-mpnet-base-v2')
    parser.add_argument('--wiki_model', type=str,
                        choices=('', 'bert-base-uncased', 'sentence-transformers/all-MiniLM-L6-v2', 'sentence-transformers/all-mpnet-base-v2'),
                        default='bert-base-uncased',
                        help='If provided, use this model for wiki input')
    parser.add_argument('--n_layers_freeze_wiki', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--inference', type=int, default=0, help='if doing inference or not')

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    print(args)

    engine = Engine(args)
    engine.train()