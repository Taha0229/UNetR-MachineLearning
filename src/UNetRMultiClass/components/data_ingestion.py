import os
from pathlib import Path
from gdown import download
import shutil
from UNetRMultiClass import logger
from UNetRMultiClass.utils.common import get_size
from UNetRMultiClass.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config    
        
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            download(self.config.source_URL, output=self.config.local_data_file)
            logger.info(f"download completed!")
        else:
            logger.info(f"Dataset files already exists of size: {get_size(Path(self.config.local_data_file))}") 
            
    def extract_zip_file(self):
        logger.info(f"trying to extract to {os.path.join(self.config.unzip_dir, 'LaPa')}")
        if not os.path.exists(os.path.join(self.config.unzip_dir, 'LaPa')):
            logger.info(f"Starting Extraction!")
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            shutil.unpack_archive(self.config.local_data_file, unzip_path)
            logger.info(f"Extraction Completed!")
        else:
            logger.info(f"Extracted files already exists") 