import wandb
from he_engine_multiple import Engine

def sweep_train():
    # Initialize a new wandb run and retrieve the current config
    wandb.init()
    config = wandb.config

    # Mimic argparse.Namespace using a simple class
    class Args:
        pass

    args = Args()
    # Fixed parameters for all experiments
    args.data = 'vast'
    args.topic = ''  # adjust as needed
    args.model = 'sentence-transformers/all-mpnet-base-v2'
    args.wiki_model = 'sentence-transformers/all-mpnet-base-v2'
    # Use the same freezing parameter for the wiki model
    args.n_layers_freeze_wiki = config.n_layers_freeze
    args.gpu = '0'
    args.inference = 0
    args.n_workers = 4
    args.seed = 42

    # Overridden hyperparameters from the sweep
    args.lr = config.lr
    args.batch_size = config.batch_size
    args.epochs = config.epochs
    args.patience = config.patience
    args.n_layers_freeze = config.n_layers_freeze
    args.l2_reg = config.l2_reg

    print("Starting training with config:", config)
    engine = Engine(args)
    engine.train()

if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',  # Use Bayesian optimization for the sweep
        'metric': {
            'name': 'val_f1',  # Ensure your training code logs this metric appropriately
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'distribution': 'uniform',
                'min': 1e-6,
                'max': 1e-4
            },
            'batch_size': {
                'values': [8, 16, 32, 64]
            },
            'patience': {
                'values': [5, 10, 15]
            },
            'epochs': {
                'values': [25, 50, 75]
            },
            'n_layers_freeze': {
                'distribution': 'int_uniform',
                'min': 0,
                'max': 12
            },
            'l2_reg': {
                'distribution': 'uniform',
                'min': 1e-5,
                'max': 1e-3
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 1,
            'eta': 3,
            'max_iter': 50
        }
    }

    # Create the sweep and get the sweep id
    sweep_id = wandb.sweep(sweep_config, project="wiki-multiple")
    wandb.agent(sweep_id, function=sweep_train, count=100)
