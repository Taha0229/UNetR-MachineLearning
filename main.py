from UNetRMultiClass import logger
from UNetRMultiClass.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from UNetRMultiClass.pipeline.stage_02_prepare_model import PrepareModelTrainingPipeline
from UNetRMultiClass.pipeline.stage_03_training import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\n x===============x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Model"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
    obj = PrepareModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\n x===============x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
    data_ingestion = ModelTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\n x===============x")
except Exception as e:
    logger.exception(e)
    raise e