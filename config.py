from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 5,
        "lr": 1e-4,
        "seq_len": 256,
        "d_model": 512,
        "source_lang": "en",
        "target_lang": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_path": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)