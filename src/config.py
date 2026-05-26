config_mnist = {
    'img_size': 28,
    'patch_size': 4,
    'num_of_channels': 1,
    'embed_dim': 16,
    'enc_depth': 4,
    'dec_depth': 2,
    'num_heads': 2,
    'mlp_dim': 64,
    'epochs_phase1': 20,
    'epochs_phase2': 500,
    'epochs_phase3': 100,
    'lr': 0.0005,
    'grow_after_epochs': 10,
    'som_rows': 10,
    'som_cols': 10,
    'stop_growth_purity': 0.90
}

config_fashion_mnist = {
    'img_size': 28,
    'patch_size': 4,
    'num_of_channels': 1,
    'embed_dim': 16,
    'enc_depth': 4,
    'dec_depth': 2,
    'num_heads': 2,
    'mlp_dim': 64,
    'epochs_phase1': 20,
    'epochs_phase2': 500,
    'epochs_phase3': 100,
    'lr': 0.0005,
    'grow_after_epochs': 10,
    'som_rows': 10,
    'som_cols': 10,
    'stop_growth_purity': 0.80
}

config_usps = {
    'img_size': 16,
    'patch_size': 4,
    'num_of_channels': 1,
    'embed_dim': 16,
    'enc_depth': 4,
    'dec_depth': 2,
    'num_heads': 2,
    'mlp_dim': 64,
    'epochs_phase1': 20,
    'epochs_phase2': 500,
    'epochs_phase3': 100,
    'lr': 0.0005,
    'grow_after_epochs': 10,
    'som_rows': 5,
    'som_cols': 5,
    'stop_growth_purity': 0.90
}