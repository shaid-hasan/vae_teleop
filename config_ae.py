
# ## Auto Encoder Sweep
# sweep_config = {
#     'method': 'grid',
#     'metric': {'name': 'loss', 'goal': 'minimize'},
#     'parameters': {
#         'hidden_size': {'values': [32, 64]},
#         'latent_dim': {'values': [8, 12, 24]},
#         'chunk_size': {'values': [10, 50, 250]},
#         'epochs': {'value': 100},
#         'learning_rate': {'value': 0.001},
#         'batch_size': {'value': 10}
#     }
# }

# Auto Encoder Testing
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'loss', 'goal': 'minimize'},
    'parameters': {
        'hidden_size': {'value': 64},
        'latent_dim': {'value': 24},
        'chunk_size': {'value': 5},        ## 10, 25, 
        'epochs': {'value': 15},
        'learning_rate': {'value': 1e-3},
        'batch_size': {'value': 10}
    }
}

# Auto Encoder Testing
static_config = {
    'input_size': 8,
    'num_layers': 2,
    'saved_model_path': '/scratch/qmz9mg/vae/results/auto_encoder/saved_models/',
    'z_tsne_path': '/scratch/qmz9mg/vae/results/auto_encoder/z_tsne/',
    'train_loss_path': '/scratch/qmz9mg/vae/results/auto_encoder/train_loss/',
}