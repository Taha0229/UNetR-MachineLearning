from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Entity for stage 01

    This is a sample entity for data ingestion
    primarily, this is the return type of the Data ingestion
    this is same as defined in the config yaml
    """

    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareModelConfig:
    """
    Entity for stage 02
    """

    root_dir: Path
    model_path: Path

    params_image_size: int
    params_num_classes: int
    params_num_layers: int
    params_hidden_dim: int
    params_mlp_dim: int
    params_num_heads: int
    params_dropout_rate: float
    params_num_patches: int
    params_patch_size: int
    params_num_channels: int
    params_learning_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    """
    Entity for stage 03 - training
    """

    root_dir: Path
    trained_model_path: Path
    untrained_model_path: Path
    training_data: Path

    params_image_size: int
    params_num_classes: int
    params_num_channels: int
    params_num_patches: int
    params_patch_size: int
    params_flat_patches_shape: tuple
    params_batch_size: int
    params_epochs: int


@dataclass(frozen=True)
class PrepareCallbacksConfig:
    """
    Entity for stage 03 - callbacks
    """

    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path
    csv_log_filepath: Path
