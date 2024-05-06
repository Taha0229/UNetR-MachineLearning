import os
from pathlib import Path
from UNetRMultiClass.constants import *
from UNetRMultiClass.utils.common import read_yaml, create_directories 
from UNetRMultiClass.entity.config_entity import (DataIngestionConfig, PrepareModelConfig)


class ConfigurationManager:
    
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        self.params.LITE_NUM_PATCHES = (self.params.LITE_IMAGE_SIZE**2)//(self.params.LITE_PATCH_SIZE**2)
        self.params.FULL_NUM_PATCHES = (self.params.FULL_IMAGE_SIZE**2)//(self.params.FULL_PATCH_SIZE**2)
        ## altering the num_patches because the YAML outputs the string instead of the expression
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
        
        
    def get_prepare_full_model_config(self) -> PrepareModelConfig:
        config = self.config.prepare_models

        create_directories([config.root_dir])

        prepare_full_model_config = PrepareModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.full_model_path),
            params_image_size=self.params.FULL_IMAGE_SIZE,
            params_num_classes=self.params.NUM_CLASSES,
            params_num_layers=self.params.FULL_NUM_LAYERS,
            params_hidden_dim=self.params.FULL_HIDDEN_DIM,
            params_mlp_dim=self.params.FULL_MLP_DIM,
            params_num_heads=self.params.FULL_NUM_HEADS,
            params_dropout_rate=self.params.DROPOUT_RATE,
            params_num_patches=self.params.FULL_NUM_PATCHES,
            params_patch_size=self.params.FULL_PATCH_SIZE,
            params_num_channels=self.params.NUM_CHANNELS,
            params_learning_rate=self.params.LEARNING_RATE,
        )

        return prepare_full_model_config

    def get_prepare_lite_model_config(self) -> PrepareModelConfig:
        config = self.config.prepare_models

        create_directories([config.root_dir])

        prepare_lite_model_config = PrepareModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.lite_model_path),
            params_image_size=self.params.LITE_IMAGE_SIZE,
            params_num_classes=self.params.NUM_CLASSES,
            params_num_layers=self.params.LITE_NUM_LAYERS,
            params_hidden_dim=self.params.LITE_HIDDEN_DIM,
            params_mlp_dim=self.params.LITE_MLP_DIM,
            params_num_heads=self.params.LITE_NUM_HEADS,
            params_dropout_rate=self.params.DROPOUT_RATE,
            params_num_patches=self.params.LITE_NUM_PATCHES,
            params_patch_size=self.params.LITE_PATCH_SIZE,
            params_num_channels=self.params.NUM_CHANNELS,
            params_learning_rate=self.params.LEARNING_RATE,
        )

        return prepare_lite_model_config