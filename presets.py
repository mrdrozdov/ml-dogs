
default_train_config = {
    'lr': 0.002,
    'max_epochs': 5,
}


default_model_config = {
    'name': 'net',
}


default_train_data_config = {
    'batch_size': 16,
    'num_workers': 4,
    'metadata_path': './data/train_list.mat',
    'images_folder': './data/Images',
}


default_eval_data_config = {
    'batch_size': 16,
    'num_workers': 4,
    'metadata_path': './data/test_list.mat',
    'images_folder': './data/Images',
}
