import os
from UNetRMultiClass.config.configuration import ConfigurationManager
from UNetRMultiClass.components.prepare_callbacks import PrepareCallback
from UNetRMultiClass.components.training import Training
from UNetRMultiClass import logger


STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_callbacks_list()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train(callback_list=callback_list)


if __name__ == "__main__":
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
