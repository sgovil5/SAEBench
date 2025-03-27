import json
import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

import sae_bench.custom_saes.base_sae as base_sae


class WTASAE(base_sae.BaseSAE):
    threshold: torch.Tensor
    sparsity_rate: float

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        sparsity_rate: float,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        assert 0 < sparsity_rate <= 1.0, f"sparsity_rate must be in (0, 1], got {sparsity_rate}"
        self.sparsity_rate = sparsity_rate

        # WTA also uses a threshold during inference
        self.use_threshold = True
        self.register_buffer(
            "threshold", torch.tensor(-1.0, dtype=dtype, device=device)
        )

    def encode(self, x: torch.Tensor):
        """Note: x can be either shape (B, F) or (B, L, F)"""
        # Get all pre-ReLU activations
        pre_relu_acts = (x - self.b_dec) @ self.W_enc + self.b_enc
        post_relu_acts = nn.functional.relu(pre_relu_acts)

        # If using threshold for inference
        if self.use_threshold and self.threshold > 0:
            encoded_acts = post_relu_acts * (post_relu_acts > self.threshold)
            return encoded_acts

        # Apply winner-takes-all (WTA) mechanism
        batch_size = x.size(0)
        k_per_feature = max(1, int(batch_size * self.sparsity_rate))
        
        # Transpose so each row corresponds to one dictionary element
        acts_t = post_relu_acts.transpose(-1, -2)  # (..., d_sae, B/BL)
        
        # Get top-k values for each feature
        topk_values, _ = acts_t.topk(k_per_feature, dim=-1)
        
        # Create masks for winners (values that meet or exceed the kth largest value)
        thresholds = topk_values[..., -1:] # (..., d_sae, 1)
        mask = (acts_t >= thresholds).float()
        
        # Apply mask and transpose back
        wta_acts = (acts_t * mask).transpose(-1, -2)  # (..., B/BL, d_sae)
        
        return wta_acts

    def decode(self, feature_acts: torch.Tensor):
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_dictionary_learning_wta_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: int | None = None,
    local_dir: str = "downloaded_saes",
) -> WTASAE:
    assert "ae.pt" in filename
    
    # Create a repo-specific directory
    repo_dir = os.path.join(local_dir, repo_id.replace("/", "_"))
    
    # Print for debugging
    print(f"Loading SAE from repo: {repo_id}, filename: {filename}")
    print(f"Using local directory: {repo_dir}")

    # Use repo_dir as the local_dir parameter
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=repo_dir,  # Use repo_dir instead of local_dir
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    if filename == "ae.pt":
        config_filename = "config.json"
    else:
        config_filename = filename.replace("ae.pt", "config.json")
    print(f"Looking for config file: {config_filename}")
    
    # Use repo_dir as the local_dir parameter
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=repo_dir,  # Use repo_dir instead of local_dir
    )

    with open(path_to_config) as f:
        config = json.load(f)

    if layer is not None:
        assert layer == config["trainer"]["layer"]
    else:
        layer = config["trainer"]["layer"]

    # Transformer lens often uses a shortened model name
    assert model_name in config["trainer"]["lm_name"]

    sparsity_rate = config["trainer"]["sparsity_rate"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "b_dec": "b_dec",
        "threshold": "threshold",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = WTASAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        sparsity_rate=sparsity_rate,
        model_name=model_name,
        hook_layer=layer,  # type: ignore
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(renamed_params)

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    if config["trainer"]["trainer_class"] == "WTATrainer":
        sae.cfg.architecture = "wta"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer']['trainer_class']}")

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder vectors are not normalized. Please normalize them")

    return sae


if __name__ == "__main__":
    # Example usage
    repo_id = "sgovil5/wtasae"
    filename = "ae.pt"
    layer = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    model_name = "EleutherAI/pythia-160m-deduped"
    hook_name = f"blocks.{layer}.hook_resid_post"

    sae = load_dictionary_learning_wta_sae(
        repo_id,
        filename,
        model_name,
        device,  # type: ignore
        dtype,
        layer=layer,
    )
    sae.test_sae(model_name)