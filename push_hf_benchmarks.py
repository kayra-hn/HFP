from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "kayrahan35/HFP-O1-Memory-Model"

print("Uploading to HuggingFace...")

# Upload HF_MODEL_CARD.md as README.md
api.upload_file(
    path_or_fileobj="HF_MODEL_CARD.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Update Model Card with 124M VRAM and Quality (Perplexity) Benchmarks"
)

# Upload benchmark_results_gpu.png
if os.path.exists("benchmark_results_gpu.png"):
    api.upload_file(
        path_or_fileobj="benchmark_results_gpu.png",
        path_in_repo="benchmark_results_gpu.png",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update 124M VRAM benchmark graph"
    )

# Upload benchmark_quality_results.png
if os.path.exists("benchmark_quality_results.png"):
    api.upload_file(
        path_or_fileobj="benchmark_quality_results.png",
        path_in_repo="benchmark_quality_results.png",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add Quality (Perplexity) benchmark graph"
    )

print("Hugging Face Hub updated successfully!")
