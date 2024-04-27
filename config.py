


# # Semi Adversarial Auto Encoder (GRU) Main Run
# sweep_config = {
#     'method': 'grid',
#     'metric': {'name': 'loss', 'goal': 'minimize'},
#     'parameters': {
#         'chunk_size': {'value': 25},                     ## 50,100
#         'latent_dim_z': {'value': 24},                  ## 24
#         'cat_dim': {'value': 24},                       ## 24

#         'epochs': {'value': 100},
#         'gen_lr': {'values': [1e-3, 1e-5]},             ## 1e-3, 1e-5
#         'reg_lr': {'values': [1e-3, 1e-5]},             ## 1e-3, 1e-5
#         'reg_lr_cat': {'values': [1e-3, 1e-5]},         ## 1e-3, 1e-5

#         'batch_size': {'value': 10},
#         'hidden_size': {'value': 64}
#     }
# }

## Semi Adversarial Auto Encoder (GRU) Testing
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'loss', 'goal': 'minimize'},
    'parameters': {
        'chunk_size': {'value': 10},
        'latent_dim_z': {'value': 24},            ## [8, 24]
        'cat_dim': {'value': 24},                 ## [8, 24]

        'gen_lr': {'value': 1e-5},
        'reg_lr': {'value': 1e-5},
        'reg_lr_cat': {'value': 1e-5},

        'epochs': {'value': 80},
        'batch_size': {'value': 10},
        'hidden_size': {'value': 64}
    }
}


## Semi Adversarial Auto Encoder (GRU) Testing
static_config = {
    'input_size': 8,
    'num_layers': 2,
    'latent_z_tsne_path': '/scratch/qmz9mg/vae/results/semi_adv_ae_gru/z_tsne/',
    'cat_dim_tsne_path': '/scratch/qmz9mg/vae/results/semi_adv_ae_gru/cat_tsne/',
    'train_loss_path': '/scratch/qmz9mg/vae/results/semi_adv_ae_gru/train_loss/',
    'saved_model_path': '/scratch/qmz9mg/vae/results/semi_adv_ae_gru/models/'
}
