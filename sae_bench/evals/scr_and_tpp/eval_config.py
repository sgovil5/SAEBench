from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import BaseEvalConfig


@dataclass
class ScrAndTppEvalConfig(BaseEvalConfig):
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="random seed",
    )

    dataset_names: list[str] = Field(
        default_factory=lambda: [
            "LabHC/bias_in_bios_class_set1",
            "canrager/amazon_reviews_mcauley_1and5",
        ],
        title="Dataset Names",
        description="List of dataset names for both the SCR and TPP metrics",
    )

    perform_scr: bool = Field(
        default=True,
        title="Perform Spurious Correlation Removal",
        description="If True, the eval will be Spurious Correlation Removal (SCR). If False, the eval will be TPP.",
    )

    early_stopping_patience: int = Field(
        default=20,
        title="Early Stopping Patience",
        description="We set early stopping patience to probe epochs, so we always train for the same amount.",
    )

    # Load datset and probes
    train_set_size: int = Field(
        default=4000,
        title="Train Set Size",
        description="Train set size for each linear probe.",
    )
    test_set_size: int = Field(
        default=1000,
        title="Test Set Size",
        description="Test set size for each linear probe.",
    )

    context_length: int = Field(
        default=128,
        title="LLM Context Length",
        description="The maximum length of each input to the LLM. Any longer inputs will be truncated, keeping only the beginning.",
    )
    probe_train_batch_size: int = Field(
        default=16,
        title="Probe Train Batch Size",
        description="DO NOT CHANGE without reading the paper appendix Section 1. The probe's train batch size effects the size of the spuriour correlation learned by the probe.",
    )

    @field_validator("probe_test_batch_size")
    def ensure_min_probe_test_batch_size(cls, value: int) -> int:
        return min(value, 500)

    probe_test_batch_size: int = Field(
        default=500,
        title="Probe Test Batch Size",
        description="Batch size when testing the linear probe",
    )
    probe_epochs: int = Field(
        default=20,
        title="Probe Epochs",
        description="Number of epochs to train the linear probe. Many epochs are needed to decrease randomness in the SCR results.",
    )
    probe_lr: float = Field(
        default=1e-3, title="Probe LR", description="Probe learning rate."
    )
    probe_l1_penalty: float = Field(
        default=1e-3,
        title="Probe L1 Penalty",
        description="L1 sparsity penalty when training the linear probe.",
    )

    sae_batch_size: int = Field(
        default=125,
        title="SAE Batch Size",
        description="SAE Batch size, inference only",
    )
    llm_batch_size: int = Field(
        default=None,
        title="LLM Batch Size",
        description="LLM batch size. This is set by default in the main script, or it can be set with a command line argument.",
    )  # type: ignore
    llm_dtype: str = Field(
        default="",
        title="LLM Data Type",
        description="LLM data type. This is set by default in the main script, or it can be set with a command line argument.",
    )
    lower_vram_usage: bool = Field(
        default=False,
        title="Lower Memory Usage",
        description="Lower GPU memory usage by moving model to CPU when not required. Will be slower and require more system memory.",
    )

    model_name: str = Field(
        default="",
        title="Model Name",
        description="Model name. Must be set with a command line argument.",
    )

    n_values: list[int] = Field(
        default_factory=lambda: [2, 5, 10, 20, 50, 100, 500],
        title="N Values",
        description="N represents the number of features we zero ablate when performing SCR or TPP. We iterate over all values of N.",
    )

    column1_vals_lookup: dict[str, list[tuple[str, str]]] = Field(
        default_factory=lambda: {
            "LabHC/bias_in_bios_class_set1": [
                ("professor", "nurse"),
                ("architect", "journalist"),
                ("surgeon", "psychologist"),
                ("attorney", "teacher"),
            ],
            "canrager/amazon_reviews_mcauley_1and5": [
                ("Books", "CDs_and_Vinyl"),
                ("Software", "Electronics"),
                ("Pet_Supplies", "Office_Products"),
                ("Industrial_and_Scientific", "Toys_and_Games"),
            ],
        },
        title="Column 1 Values Lookup",
        description="Column1 Values apply only to the SCR metric. Column1 values represents the class pairs we train the linear probes on. In each case, we will create a perfectly biased dataset, such as all professors are males and all nurses are females.",
    )
    
    cache_dir: str | None = None
