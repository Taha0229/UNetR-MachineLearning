import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from pathlib import Path
from patchify import patchify
from UNetRMultiClass import logger
from UNetRMultiClass.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.untrained_model_path
        )
    
    def load_dataset(self, data_path):
        ## path should be -> artifacts/LaPa
        train_x = sorted(glob(os.path.join(data_path, "train", "images", "*.jpg")))
        train_y = sorted(glob(os.path.join(data_path, "train", "labels", "*.png")))

        valid_x = sorted(glob(os.path.join(data_path, "val", "images", "*.jpg")))
        valid_y = sorted(glob(os.path.join(data_path, "val", "labels", "*.png")))

        test_x = sorted(glob(os.path.join(data_path, "test", "images", "*.jpg")))
        test_y = sorted(glob(os.path.join(data_path, "test", "labels", "*.png")))

        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

    def read_image(self, path):
        path = path.decode()
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.config.params_image_size, self.config.params_image_size))
        image = image / 255.0

        """ Processing to patches """
        patch_shape = (self.config.params_patch_size, self.config.params_patch_size, self.config.params_num_channels)
        patches = patchify(image, patch_shape, self.config.params_patch_size)
        patches = np.reshape(patches, self.config.params_flat_patches_shape)
        patches = patches.astype(np.float32)

        return patches

    def read_mask(self, path):
        path = path.decode()
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.config.params_image_size, self.config.params_image_size))
        mask = mask.astype(np.int32)
        return mask

    def tf_parse(self, x, y):
        def _parse(x, y):
            x = self.read_image(x)
            y = self.read_mask(y)
            y = tf.one_hot(y, self.config.params_num_classes)
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
        x.set_shape(self.config.params_flat_patches_shape)
        y.set_shape([self.config.params_image_size, self.config.params_image_size, self.config.params_num_classes])
        return x, y

    def tf_dataset(self, X, Y, batch=2):
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        ds = ds.map(self.tf_parse).batch(batch).prefetch(10)
        return ds




    def train(self, callback_list: list):
        # self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        # self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        np.random.seed(42)
        tf.random.set_seed(42)
        
        rgb_codes = [
        [0, 0, 0], [0, 153, 255], [102, 255, 153], [0, 204, 153],
        [255, 255, 102], [255, 255, 204], [255, 153, 0], [255, 102, 255],
        [102, 0, 51], [255, 204, 255], [255, 0, 102]
        ]

        classes = [
        "background", "skin", "left eyebrow", "right eyebrow",
        "left eye", "right eye", "nose", "upper lip", "inner mouth",
        "lower lip", "hair"
        ]
        
        dataset_path = self.config.training_data
        print("dataset_path: ", dataset_path)
        print("dataset_path type: ", type(dataset_path))
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = self.load_dataset(dataset_path)
        
        logger.info(f"Train: \t{len(train_x)} - {len(train_y)}")
        logger.info(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
        logger.info(f"Test: \t{len(test_x)} - {len(test_y)}")
 

        train_dataset = self.tf_dataset(train_x, train_y, batch=self.config.params_batch_size)
        valid_dataset = self.tf_dataset(valid_x, valid_y, batch=self.config.params_batch_size)
        
        self.model.fit(
            train_dataset,
            epochs=self.config.params_epochs,
            validation_data=valid_dataset,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)