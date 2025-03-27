import json
import os

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.custom_saes.batch_topk_sae as batch_topk_sae
import sae_bench.custom_saes.gated_sae as gated_sae
import sae_bench.custom_saes.jumprelu_sae as jumprelu_sae
import sae_bench.custom_saes.relu_sae as relu_sae
import sae_bench.custom_saes.topk_sae as topk_sae
import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.autointerp.main as autointerp
import sae_bench.evals.core.main as core
import sae_bench.evals.ravel.main as ravel
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.unlearning.main as unlearning
import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.custom_saes.wta_sae as wta_sae

MODEL_CONFIGS = {
    "pythia-70m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
        "layers": [3],
        "d_model": 512,
    },
    "pythia-410m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
        "layers": [3],
        "d_model": 1024,
    },
    # "pythia-160m-deduped": {
    #     "batch_size": 256,
    #     "dtype": "float32",
    #     "layers": [8],
    #     "d_model": 768,
    # },
    # "gemma-2-2b": {
    #     "batch_size": 32,
    #     "dtype": "bfloat16",
    #     "layers": [5, 12, 19],
    #     "d_model": 2304,
    # },
}

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "unlearning": "eval_results/unlearning",
    "ravel": "eval_results/ravel",
}


TRAINER_LOADERS = {
    # "MatryoshkaBatchTopKTrainer": batch_topk_sae.load_dictionary_learning_matryoshka_batch_topk_sae,
    "BatchTopKTrainer": batch_topk_sae.load_dictionary_learning_batch_topk_sae,
    # "TopKTrainer": topk_sae.load_dictionary_learning_topk_sae,
    # "StandardTrainerAprilUpdate": relu_sae.load_dictionary_learning_relu_sae,
    # "StandardTrainer": relu_sae.load_dictionary_learning_relu_sae,
    # "PAnnealTrainer": relu_sae.load_dictionary_learning_relu_sae,
    # "JumpReluTrainer": jumprelu_sae.load_dictionary_learning_jump_relu_sae,
    # "GatedSAETrainer": gated_sae.load_dictionary_learning_gated_sae,
    "WTATrainer": wta_sae.load_dictionary_learning_wta_sae,
}


def get_all_hf_repo_autoencoders(
    repo_id: str, download_location: str = "downloaded_saes"
) -> list[str]:
    download_location = os.path.join(download_location, repo_id.replace("/", "_"))
    config_dir = snapshot_download(
        repo_id,
        allow_patterns=["*config.json"],
        local_dir=download_location,
        force_download=False,
    )

    config_locations = []

    for root, _, files in os.walk(config_dir):
        for file in files:
            if file == "config.json":
                config_locations.append(os.path.join(root, file))

    # Print for debugging
    print(f"Found config locations: {config_locations}")
    
    repo_locations = []

    for config in config_locations:
        # Extract the relative path from the download location
        rel_path = os.path.relpath(os.path.dirname(config), download_location)
        
        # If the config is in the root directory, use empty string
        if rel_path == ".":
            repo_locations.append("")
        else:
            repo_locations.append(rel_path)
    
    # Print for debugging
    print(f"Extracted repo locations: {repo_locations}")
    
    return repo_locations


def load_dictionary_learning_sae(
    repo_id: str,
    location: str,
    model_name,
    device: str,
    dtype: torch.dtype,
    layer: int | None = None,
    download_location: str = "downloaded_saes",
) -> base_sae.BaseSAE:
    download_location = os.path.join(download_location, repo_id.replace("/", "_"))
    
    # Fix: Handle the case where location is empty (root directory)
    if location == "":
        config_file = os.path.join(download_location, "config.json")
        ae_path = "ae.pt"
    else:
        config_file = os.path.join(download_location, location, "config.json")
        ae_path = os.path.join(location, "ae.pt")
    
    # Print for debugging
    print(f"Looking for config at: {config_file}")
    
    with open(config_file) as f:
        config = json.load(f)

    trainer_class = config["trainer"]["trainer_class"]
    
    # Print for debugging
    print(f"Using trainer class: {trainer_class}")
    print(f"Loading ae.pt from: {ae_path}")

    sae = TRAINER_LOADERS[trainer_class](
        repo_id=repo_id,
        filename=ae_path,
        layer=layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )
    return sae


def verify_saes_load(
    repo_id: str,
    sae_locations: list[str],
    model_name: str,
    device: str,
    dtype: torch.dtype,
):
    """Verify that all SAEs load correctly. Useful to check this before a big evaluation run."""
    for sae_location in sae_locations:
        sae = load_dictionary_learning_sae(
            repo_id=repo_id,
            location=sae_location,
            layer=None,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
        del sae


def run_evals(
    repo_id: str,
    model_name: str,
    sae_locations: list[str],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    random_seed: int,
    api_key: str | None = None,
    force_rerun: bool = True,
    cache_dir: str | None = None,
):
    """Run selected evaluations for the given model and SAEs."""

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    # Mapping of eval types to their functions and output paths
    eval_runners = {
        "absorption": (
            lambda selected_saes, is_final: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    cache_dir=cache_dir,
                ),
                selected_saes,
                device,
                "eval_results/absorption",
                force_rerun,
            )
        ),
        "autointerp": (
            lambda selected_saes, is_final: autointerp.run_eval(
                autointerp.AutoInterpEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                api_key,  # type: ignore
                "eval_results/autointerp",
                force_rerun,
            )
        ),
        # TODO: Do a better job of setting num_batches and batch size
        # The core run_eval() interface isn't well suited for custom SAEs, so we have to do this instead.
        # It isn't ideal, but it works.
        # TODO: Don't hardcode magic numbers
        "core": (
            lambda selected_saes, is_final: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder="eval_results/core",
                verbose=True,
                dtype=llm_dtype,
                device=device,
            )
        ),
        "ravel": (
            lambda selected_saes, is_final: ravel.run_eval(
                ravel.RAVELEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size // 4,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/ravel",
                force_rerun,
            )
        ),
        "scr": (
            lambda selected_saes, is_final: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    perform_scr=True,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    cache_dir=cache_dir,
                ),
                selected_saes,
                device,
                "eval_results",  # We add scr or tpp depending on perform_scr
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "tpp": (
            lambda selected_saes, is_final: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    perform_scr=False,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    cache_dir=cache_dir,
                ),
                selected_saes,
                device,
                "eval_results",  # We add scr or tpp depending on perform_scr
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "sparse_probing": (
            lambda selected_saes, is_final: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/sparse_probing",
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "unlearning": (
            lambda selected_saes, is_final: unlearning.run_eval(
                unlearning.UnlearningEvalConfig(
                    model_name="gemma-2-2b-it",
                    random_seed=random_seed,
                    llm_dtype=llm_dtype,
                    llm_batch_size=llm_batch_size
                    // 8,  # 8x smaller batch size for unlearning due to longer sequences
                ),
                selected_saes,
                device,
                "eval_results/unlearning",
                force_rerun,
            )
        ),
    }

    for eval_type in eval_types:
        if eval_type not in eval_runners:
            raise ValueError(f"Unsupported eval type: {eval_type}")

    verify_saes_load(
        repo_id,
        sae_locations,
        model_name,
        device,
        general_utils.str_to_dtype(llm_dtype),
    )

    # Run selected evaluations
    for eval_type in tqdm(eval_types, desc="Evaluations"):
        if eval_type == "autointerp" and api_key is None:
            print("Skipping autointerp evaluation due to missing API key")
            continue
        if eval_type == "unlearning":
            if not os.path.exists(
                "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
            ):
                print(
                    "Skipping unlearning evaluation due to missing bio-forget-corpus.jsonl"
                )
                continue

        print(f"\n\n\nRunning {eval_type} evaluation\n\n\n")

        try:
            for i, sae_location in enumerate(sae_locations):
                is_final = False
                if i == len(sae_locations) - 1:
                    is_final = True

                sae = load_dictionary_learning_sae(
                    repo_id=repo_id,
                    location=sae_location,
                    layer=None,
                    model_name=model_name,
                    device=device,
                    dtype=general_utils.str_to_dtype(llm_dtype),
                )
                unique_sae_id = sae_location.replace("/", "_")
                unique_sae_id = f"{repo_id.split('/')[1]}_{unique_sae_id}"
                selected_saes = [(unique_sae_id, sae)]

                os.makedirs(output_folders[eval_type], exist_ok=True)
                eval_runners[eval_type](selected_saes, is_final)

                del sae

        except Exception as e:
            print(f"Error running {eval_type} evaluation: {e}")
            continue


if __name__ == "__main__":
    """
    This will run all evaluations on all selected dictionary_learning SAEs within the specified HuggingFace repos.
    Set the model_name(s) and repo_id(s) in `repos`.
    Also specify the eval types you want to run in `eval_types`.
    You can also specify any keywords to exclude/include in the SAE filenames using `exclude_keywords` and `include_keywords`.
    NOTE: If your model (with associated model_name and batch sizes) is not in the MODEL_CONFIGS dictionary, you will need to add it.
    This relies on each SAE being located in a folder which contains an ae.pt file and a config.json file (which is the default save format in dictionary_learning).
    Running this script as is should run SAE Bench Pythia and Gemma SAEs.
    """
    RANDOM_SEED = 42
    
    scratch_dir = "/storage/home/hcoda1/0/sgovil9/scratch"
    os.makedirs(scratch_dir, exist_ok=True)
    
    hf_cache_dir = os.path.join(scratch_dir, "hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")
    
    # Set Weights & Biases cache directory
    wandb_cache_dir = os.path.join(scratch_dir, "wandb")
    os.makedirs(wandb_cache_dir, exist_ok=True)
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
    
    # Set download location for SAEs
    download_location = os.path.join(scratch_dir, "downloaded_saes")
    os.makedirs(download_location, exist_ok=True)

    device = general_utils.setup_environment()

    # Select your eval types here.
    # Note: Unlearning is not recommended for models with < 2B parameters and we recommend an instruct tuned model
    # Unlearning will also require requesting permission for the WMDP dataset (see unlearning/README.md)
    # Absorption not recommended for models < 2B parameters
    eval_types = [
        # "absorption",
        "core",
        "scr",
        "tpp",
        "sparse_probing",
        # "autointerp",
        # "unlearning",
        "ravel",
    ]

    if "autointerp" in eval_types:
        try:
            with open("openai_api_key.txt") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise Exception("Please create openai_api_key.txt with your API key")
    else:
        api_key = None

    if "unlearning" in eval_types:
        if not os.path.exists(
            "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
        ):
            raise Exception(
                "Please download bio-forget-corpus.jsonl for unlearning evaluation"
            )

    repos = [
        (
            "sgovil5/wtasae",
            "pythia-410m-deduped",
        ),
        (
            "sgovil5/batchtopksae",
            "pythia-410m-deduped",
        )
    ]
    exclude_keywords = ["checkpoints"]
    include_keywords = []

    for repo_id, model_name in repos:
        print(f"\n\n\nEvaluating {model_name} with {repo_id}\n\n\n")

        llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
        str_dtype = MODEL_CONFIGS[model_name]["dtype"]
        torch_dtype = general_utils.str_to_dtype(str_dtype)

        sae_locations = get_all_hf_repo_autoencoders(repo_id)

        sae_locations = general_utils.filter_keywords(
            sae_locations,
            exclude_keywords=exclude_keywords,
            include_keywords=include_keywords,
        )

        run_evals(
            repo_id=repo_id,
            model_name=model_name,
            sae_locations=sae_locations,
            llm_batch_size=llm_batch_size,
            llm_dtype=str_dtype,
            device=device,
            eval_types=eval_types,
            api_key=api_key,
            random_seed=RANDOM_SEED,
            force_rerun=True
        )
