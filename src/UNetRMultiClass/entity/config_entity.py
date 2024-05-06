from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    This is a sample entity for data ingestion
    primarily, this is the return type of the Data ingestion
    this is same as defined in the config yaml
    """

    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
