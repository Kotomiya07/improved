from huggingface_hub import snapshot_download

model_names = [
    "Kotomiya07/kl-f2",
    "Kotomiya07/kl-f4",
    "Kotomiya07/vq-f4",
    "Kotomiya07/vq-f8",
]

download_path -""

for model_name in model_names:
    snapshot_download(
        repo_id=model_name,
        local_dir = f"../autoencoder/{model_name}",
        local_dir_use_symlinks=False # ※1
        )
