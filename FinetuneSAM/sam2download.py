from huggingface_hub import hf_hub_download

# Corrected repo_id (added the .1)
path = hf_hub_download(
    repo_id="facebook/sam2.1-hiera-base-plus", 
    filename="sam2.1_hiera_base_plus.pt",
    local_dir="./weights"
)

print(f"Success! Model downloaded to: {path}")