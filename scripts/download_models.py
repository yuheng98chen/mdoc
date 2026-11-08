from huggingface_hub import snapshot_download
import os


current_dir = os.path.dirname(os.path.abspath(__file__))


retrieval_dir = os.path.abspath(os.path.join(current_dir, "..", "retrieval"))
models_dir = os.path.abspath(os.path.join(current_dir, "..", "models"))


snapshot_download(
    repo_id="vidore/colpaligemma-3b-mix-448-base",
    local_dir=os.path.join(retrieval_dir, "local_colpaligemma"),
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="vidore/colpali",
    local_dir=os.path.join(retrieval_dir, "local_colpali_adapter"),
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="colbert-ir/colbertv2.0",
    local_dir=os.path.join(retrieval_dir, "local_colbert"),
    local_dir_use_symlinks=False
)

qwen_path7b = snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir=os.path.join(models_dir, "qwen2.5-vl-7b"),
    local_dir_use_symlinks=False,
    revision="main"
)

qwen_path32b = snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-32B-Instruct",
    local_dir=os.path.join(models_dir, "qwen2.5-vl-32b"),
    local_dir_use_symlinks=False,
    revision="main"
)

falcon_path = snapshot_download(
    repo_id="tiiuae/falcon-rw-1b",
    local_dir=os.path.join(models_dir, "falcon-rw-1b"),
    local_dir_use_symlinks=False,
    revision="main"
)

print(f"Qwen2.5 7b VL model downloaded to: {qwen_path7b}")
print(f"Qwen2.5 32b VL model downloaded to: {qwen_path32b}")
print(f"Falcon RW 1B model downloaded to: {falcon_path}")
