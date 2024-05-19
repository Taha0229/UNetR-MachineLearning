# Multi-class Image Segmentation using UNETR

⚡Combining the power of Transformers with UNet for state-of-the-art image segmentation task💪
# Project Brief

In October 2021, Ali Hatamizadeh et al. published a paper titled "UNETR: Transformers for 3D Medical Image Segmentation," introducing the UNETR architecture, which outperforms other segmentation models. In essence, UNETR utilizes a contracting-expanding pattern consisting of a stack of transformer as the encoder which is connected to the CNN-based decoder via skip connections, producing segmented image. 
<br><br>
This project aims to implement the UNETR architecture as described in the paper, training it on a custom multi-class dataset for facial feature segmentation. The project involves developing the machine learning model, backend, and frontend for the application. The UNETR model is served via a REST API using Django REST framework to a Next.js frontend, with the frontend and backend deployed separately on Vercel and AWS, respectively. This tech stack selection ensures high scalability, performance, and an excellent UI/UX.
<br><br>
The ML implementation emphasizes modular, object-oriented pipelines. These pipelines include data ingestion from remote sources, model preparation, model training, and model evaluation, all managed by DVC for streamlined MLOps.

# About UNETR

The UNETR architecture is primarily designed for 3D medical image segmentation, including MRI, CT, and ultrasound scans. However, it can also be adapted for other domains such as facial feature segmentation and self-driving vehicles.

UNETR utilizes a Vision Transformer (ViT) as the encoder to learn global contextual representations and a CNN-based decoder to up-sample these representations, generating the final segmentation mask.

### Vision Transformer 
Vision Transformer (ViT) is an architecture used for image recognition, based on the Transformer architecture initially developed for natural language processing. ViTs have achieved state-of-the-art results in various image recognition tasks, including ImageNet classification.
(insert: image)
The idea is to divide the input image into patches which are fed into the encoder block.
Position embedding provides information about the sequence of patches to better understand the overall image context, similar to how positional embedding is used in NLP transformers.
(basic architecture image)

**Patch calculation:**
Input Image = H * W * C <br>
Patch Size = Ph * Pw <br>
Number of patches (N) = (H * W)/(Ph * Pw) <br>
Transformed Input = (N, Ph * Pw * C) <br>

H = Height <br>
W = Width <br>
C = Image channels <br>
Ph = Patch height <br>
Pw = Patch width <br>
N = Number of patches 

Example:
Input Image -> 200 pixels * 200 pixels * 3 (RBG channels) <br>
Patch size = 25 * 25<br>
Number of patches (N) = (200 * 200) / (25 * 25)
                      = 64<br>
Transformed input = (64, 25*25*3)<br>
                  = (64, 1875)<br>

The ViT comes in three sizes-

| Model     | Layers | Hidden Size (D) | MLP Size | Heads | Params |
|-----------|--------|-----------------|----------|-------|--------|
| ViT-Base  | 12     | 768             | 3072     | 12    | 86M    |
| ViT-Large | 24     | 1024            | 4096     | 16    | 307M   |
| ViT-Huge  | 32     | 1280            | 5120     | 16    | 632M   |





### Complete UNETR Architecture


(insert: image)


If you are only interested in my implementation of the UNETR architecture, excluding the training pipelines, you can view `model_architecture.py` here.


# Project Implementation 

**The project is divided into three modules, consisting of**
1. ML model (link): This module involves building pipelines from data ingestion to model training. In this repository, we focus primarily on this part, providing a basic overview of the other modules.<br><br>
2. Backend + its deployment (link):  This module uses Django REST Framework to serve the model, containerizes the application using Docker, pushes the image to AWS ECR, implements CI/CD with GitHub Actions, and deploys on AWS EC2.<br><br>
3. Frontend + its deployment (link): This module involves building a Next.js app that utilizes Tailwind CSS and NextUI for a beautiful UI/UX, and deploying on Vercel.<br>

# How to Install and Run the ML Model

### Step 1: Fork and Clone the repository 
Fork the repo then head to the folder where you wish to clone the project.<br>
Open the folder using VS Code. `Right click > Open with VS Code` or on your terminal open the folder then use `code .`<br>
open your VS Code terminal using ctrl + &tilde;<br>
use the command to clone the repo:<br>
`git clone (link)`

### Step 2: Create a virtual environment
I prefer to create virtual environment using conda but you can use your favorite method. If you do not have conda you can download and install it by following (link: this).

In the VS Code's terminal use the following command:<br>
`conda create --name unetr-ml python=3.10 -y`<br>
after the installation is completed use:<br>
`conda activate unetr-ml`<br>

**note:** If you do not mention the python version, conda will not install python and you will be using the global python interpreter or virtual env. 
<br>
In some cases, even though you have mentioned the python version, VS Code doesn't select the virtual env when you activate it for the project. So, to be 100% sure we use the following command:
`pip list`<br>
This should list only a 3-5 dependencies. If a lot of dependencies are listed then you can manually select a python interpreter on VS Code. 
<br>
For this open any python file with the extension `.py`, then on the very bottom left you can see the python interpreter<br>
(insert: image)<br>
Then a popup will show up on the top, refresh it and somewhere in the list you can find `unetr` virtual env.<br>
(insert: image)


### Step 3: Install all the dependencies
In python, unlike javascript, we can not use other files/folders in our project unless we declare the whole project as a package/module, otherwise we will get `Module not found` error.<br>
For the reason we are using `setup tools` in our `setup.py` to setup the project as a package/module.<br><br>
In `setup.py`:<br>
`package_dir={"": "src"}` and `packages=setuptools.find_packages(where="src")` defines the directory of the package, therefore we can only access those files and folders which are inside the `src` folder and `e .` in   `requirements.txt` in python projects is used to indicate that the project should be installed in "editable" or "development" mode. 
<br><br>
With this we have configured all the pre-requisite for dependencies installation.
Use the following command to install all the requirements:
`pip install -r requirements.txt`

### Step 4: Inference
To test (because we will be using django to server the model in the next module) the inference, you can either import the `PredictionPipeline` class and instantiate it from `src > UNetRMultiClass > pipeline > predict.py` or instantiate the same class in `predict.py` itself. <br><br>
Example (`in predict.py`):<br>
`pred_obj = PredictionPipeline()`
`pred_obj.predict("image_name.extension")`

The output image can be viewed in the `outputs\predict` folder

### Step 5: Training (if required)
Before starting the training please read the preface (add: link) on my implementation, as there are few important aspect regarding the model training.<br>
You can either run all the pipeline by executing `main.py` with the following command on the terminal:<br>
`python main.py`<br>
or you can use the following DVC command to inspect any changes in the pipeline and only execute those pipeline if there are any changes.<br>
`dvc init`<br>
`dvc repro`

You can use `dvc dag` to view acyclic graph 

#### Understanding Tech-
DVC: is an open-source tool designed to manage machine learning projects. It facilitates tracking and versioning of data, models, and pipelines, similar to how Git handles code. It executes only those pipelines which encountered some sort of modification. For example, if a model architecture is changed then `prepare_model` pipeline will be executed with its dependent pipeline.

### Step 6: Model Deployment and Usage
In the next module (link), we will develop and deploy the backend using Django REST Framework. This includes serving the model over a REST API and deploying it on AWS EC2 running Ubuntu, with CI/CD pipelines built using GitHub Actions, Docker, and AWS ECR.

Following that, in the frontend module (link), we will develop a Next.js app with Tailwind CSS and deploy it on Vercel.

**note:** Training this model requires significant time and computational resources. It is recommended to perform training on the cloud or use free-tier resources such as Google Colab and Kaggle.

# Understanding Project Structure
The goal is to build four pipelines: data ingestion, model preparation, model training, and model evaluation. Each pipeline will consist of distinct components: Entity, Config, Component, and Pipeline, organized in a well-structured and collaborative format.

## Root Folder:
In root folder, we have:
1. dvc.yaml: defines the configuration for DVC 
2. main.py: All the pipelines are executed from main.py
3. params.yaml: Contains Hyper-parameters for the model
4. requirements.txt: Lists all the requirements along with their versions
5. Setup.py: To setup the project as a package
6. template.py: Used in the beginning of the project to create files and folders and populate them as needed


├───artifacts                                               # for storing by-products during the development process 
│   ├───data_ingestion                                      ## stores zipped dataset
│   ├───LaPa                                                ## Dataset name
│   │   ├───test                                            ## Test dataset
│   │   │   ├───images
│   │   │   ├───labels
│   │   │   └───landmarks
│   │   ├───train                                           ## Train dataset
│   │   │   ├───images
│   │   │   ├───labels
│   │   │   └───landmarks
│   │   └───val                                             ## Cross Validation/Dev dataset
│   │       ├───images
│   │       ├───labels
│   │       └───landmarks
│   ├───prepare_callbacks                                   ## Artifacts due to callbacks
│   │   ├───checkpoint_dir
│   │   ├───csv_log
│   │   └───tensorboard_log_dir
│   │       ├───tb_logs_at_2024-05-07-03-58-10              
│   │       │   └───train
│   │       ├───tb_logs_at_2024-05-07-04-02-02
│   │       │   └───train
│   │       ├───tb_logs_at_2024-05-07-04-51-04
│   │       │   └───train
│   │       └───tb_logs_at_2024-05-19-23-06-06
│   │           └───train
│   ├───prepare_model                                       ## Prepared Model Architecture which will be Training later
│   └───training                                            ## Final Trained Model
├───config                                                  ## Contains config.yaml for all the configuration related to the pipelines such as remote dataset url, local dirs, etc
├───logs                                                    ## Contain logs generated during execution of pipelines
├───outputs                                                 ## Contains Model's output
│   └───predict
├───research                                                ## Contains python notebooks for testing individual pipelines and model behavior   
│   └───logs
├───src                                                     ## Contains the core implementation
    │   │   └───__pycache__
    ├───UNetRMultiClass
    │   ├───components                  ## Contains components for each pipelines which are responsible for methods used in them. (UNETR implementation can be found in prepare_model.py)
    │   │   └───__pycache__
    │   ├───config                                          ## Contains Configuration for all the pipelines along with enforced return types from entities
    │   │   └───__pycache__
    │   ├───constants                                       ## Contains Constants used in the project
    │   │   └───__pycache__
    │   ├───entity                                          ## Contains Entities for pipeline's Configs (more on this later). 
    │   │   └───__pycache__     
    │   ├───pipeline                                        ## Contains the main pipelines 
    │   │   └───__pycache__
    │   ├───utils                                           ## Contains the commonly used utils
    │   └───__pycache__
    └───UNetRMultiClass.egg-info



### Special Mentions:
1. Entity : primarily, an entity is the return type of the pipeline's Config. This is used validate the configurations of the pipeline.
2. `src > UNetRMultiClass > __init__.py` setups logger system which can be used anywhere across the project to log anything.

## **Note:**
**Tl;dr:**
1. The model requires a lot of computation due to its massive model size and minimum dataset requirements.<br>
2. In this project I have built 2 models, which I named as - i. `full_model` and ii. `lite_model`<br>
The size of `full_model` was- 1.55 GB with 86 Million parameters <br>
The size of `lite_model` was - 22.5 MB with y parameters<br>
3. Trained the `lite_model` on Google Colab with T4 GPU<br>
4. Keras version in my local machine and on the Colab was different, therefore using `keras.load_model()` method throw incompatible error, hence I have converted the model into a onnx model (`compatible_model.onnx`)



With the base model having 86 million parameters and a total size of 1.55 GB, training it posed significant computational challenges. Additionally, the research indicated the need for a dataset with a minimum number of examples, further complicating the training process. The researchers used an Nvidia DGX-1 server, which required 10 hours to train for 20,000 epochs, even on such powerful hardware to achieve state-of-the-art results. 
Therefore, I have tweaked the model architecture by altering a, b, c resulting in a lighter model with 22.5 million parameters and a size of 22.5 MB. Despite the reduced model size, meeting the minimum dataset requirements still made training difficult. On my local machine, equipped with a Ryzen 5 3500U, 8 GB RAM, and an integrated GPU, it took 4 hours to train for just one epoch.<br><br>
For the reason being, I trained the model on Google Colab (link here) on T4 GPU which is available with the free tier. The training on Google Colab took approximately 10 minutes per epoch, and the model was trained for 10 epochs, taking a total of 1 hour and 40 minutes (exhausting my free tier). The trained weights are available in the `artifacts > training` directory.<br>

However, a version mismatch between TensorFlow on Google Colab and the version used in this project caused an internal error due to different Keras versions during model inference. To resolve this, I converted the model to a compatible ONNX format. You can find `compatible_model.onnx` in the `artifacts > training` directory as well as in this Colab notebook (Google Colab link).

If you run the project, then the `lite_model.keras` and `full_model.keras` can be found in `artifacts > prepare_model`<br>

# How to build your own project 

### Step 1: template.py
copy-paste `template.py` which creates the project structure, alter it as per your requirements.

### Step 2: setup.py
copy-paste `setup.py` and alter it as per your requirements.

### Step 3: Initial extras and configs
1. list all of your hyper-parameters in `param.yaml` if it is known prior to model developments (e.g. from research paper). 
2. Implement logger in `src > <project_name> > __init__.py`.
3. configure `config.yaml` in `config` folder

### Step 4: Create Working Pipelines
1. Create and test the pipeline in a notebook in `research` folder by following steps:<br>
1.1. Define Entity for the pipeline's config.<br>
1.2. Configure `ConfigurationManager` class by implementing methods to define the configs required for the pipeline. The return type of a particular pipeline's config will be the entity for that pipeline.<br>
1.3 Implement the component, here we define the methods which will be consumed in the pipeline<br>
1.4 Build the pipeline by instantiating the `ConfigurationManager` to get the pipeline's configs, followed by instantiating the component by passing the configs to it and finally consuming the required methods from the component. <br>

2. Follow the workflow below to implement all the above mentioned steps to create actual working pipelines.

## Workflows

1. Update `config > config.yaml`
2. Update `secrets.yaml` [Optional]
3. Update `params.yaml`
4. Update `src > <project_name> > entity > config_entity.py`
5. Update `src > <project_name> > config > configuration.py`
6. Update `src > <project_name> > component > <pipeline_name>.py`
7. Update `src > <project_name> > pipeline > stage_<x_pipeline_name>.py`
8. Update the main.py
9. Update the dvc.yaml 

### GitHub Commit message format
Feat– feature

Fix– bug fixes

Docs– changes to the documentation like README

Style– style or formatting change 

Perf – improves code performance

Test– test a feature