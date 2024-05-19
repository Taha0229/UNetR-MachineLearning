import os
import tensorflow as tf
from UNetRMultiClass import logger
import cv2
import numpy as np
from patchify import patchify
import onnxruntime as ort


class PredictionPipeline:
    def __init__(self):

        self.rgb_codes = [
            [0, 0, 0],
            [0, 153, 255],
            [102, 255, 153],
            [0, 204, 153],
            [255, 255, 102],
            [255, 255, 204],
            [255, 153, 0],
            [255, 102, 255],
            [102, 0, 51],
            [255, 204, 255],
            [255, 0, 102],
        ]

        self.classes = [
            "background",
            "skin",
            "left eyebrow",
            "right eyebrow",
            "left eye",
            "right eye",
            "nose",
            "upper lip",
            "inner mouth",
            "lower lip",
            "hair",
        ]

        self.onnx_model_path = "artifacts/training/compatible_model.onnx"
        self.session = ort.InferenceSession(self.onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def grayscale_to_rgb(self, mask, rgb_codes):
        h, w = mask.shape[0], mask.shape[1]
        mask = mask.astype(np.int32)
        output = []

        enum = enumerate(mask.flatten())

        for i, pixel in enum:
            output.append(rgb_codes[pixel])

        output = np.reshape(output, (h, w, 3))

        return output

    def save_results(self, image_x, pred, save_image_path):

        pred = np.expand_dims(pred, axis=-1)
        pred = self.grayscale_to_rgb(pred, self.rgb_codes)

        line = np.ones((image_x.shape[0], 10, 3)) * 255

        cat_images = np.concatenate([image_x, line, pred], axis=1)

        cv2.imwrite(save_image_path, cat_images)

    def predict(self, filename):
        cf = {}
        cf["image_size"] = 256
        cf["num_classes"] = 11
        cf["num_channels"] = 3
        cf["num_layers"] = 12
        cf["hidden_dim"] = 128
        cf["mlp_dim"] = 32
        cf["num_heads"] = 6
        cf["dropout_rate"] = 0.1
        cf["patch_size"] = 16
        cf["num_patches"] = (cf["image_size"] ** 2) // (cf["patch_size"] ** 2)
        cf["flat_patches_shape"] = (
            cf["num_patches"],
            cf["patch_size"] * cf["patch_size"] * cf["num_channels"],
        )

        image_name = filename
        display_name = image_name.split("/")[-1].split(".")[0]
        logger.info(f"input_image name is: {display_name}")

        input_img = cv2.imread(image_name, cv2.IMREAD_COLOR)
        input_img = cv2.resize(input_img, (cf["image_size"], cf["image_size"]))
        norm_input_img = input_img / 255.0

        patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
        patches = patchify(norm_input_img, patch_shape, cf["patch_size"])
        patches = np.reshape(patches, cf["flat_patches_shape"])
        patches = patches.astype(np.float32)  # [...]
        patches = np.expand_dims(patches, axis=0)  # [1, ...]

        """ Prediction """

        input_dict = {self.input_name: patches}
        outputs = self.session.run([self.output_name], input_dict)
        pred_1 = np.argmax(outputs, axis=-1)  ## [0.1, 0.2, 0.1, 0.6] -> 3
        pred_1 = pred_1.astype(np.int32)
        pred_1 = np.reshape(pred_1, (256, 256))

        print("saving")
        save_image_path = f"outputs/predict/{display_name}.png"
        self.save_results(input_img, pred_1, save_image_path)

        return save_image_path
