from UNetRMultiClass.config.configuration import ConfigurationManager
from UNetRMultiClass.components.prepare_model import PrepareModel
from UNetRMultiClass import logger
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

STAGE_NAME = "Prepare Model"

class PrepareModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_full_model_config = config.get_prepare_full_model_config()
        if not os.path.exists(prepare_full_model_config.model_path):
            prepare_full_model = PrepareModel(config=prepare_full_model_config)
            prepare_full_model.get_full_model()
        else:
            logger.info("Full model already exists, skipping ")
            
        prepare_lite_model_config = config.get_prepare_lite_model_config()
        if not os.path.exists(prepare_lite_model_config.model_path):
            prepare_lite_model = PrepareModel(config=prepare_lite_model_config)
            prepare_lite_model.get_lite_model()
        else:
            logger.info("Lite  model already exists, skipping ")
    
    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n x===============x")
    except Exception as e:
        logger.exception(e)
        raise e