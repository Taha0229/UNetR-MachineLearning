from UNetRMultiClass.entity.config_entity import PrepareCallbacksConfig
import time
import os
import tensorflow as tf


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            verbose=1,
            save_best_only=True,
        )

    @property
    def _create_csvlogger_callbacks(self):
        return tf.keras.callbacks.CSVLogger(self.config.csv_log_filepath)

    @property
    def _create_reducelr_callbacks(self):
        return (
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7, verbose=1
            ),
        )

    @property
    def _create_earlystopping_callbacks(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=False
        )

    def get_callbacks_list(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks,
            self._create_csvlogger_callbacks,
            self._create_reducelr_callbacks,
            self._create_earlystopping_callbacks,
        ]
