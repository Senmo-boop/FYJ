                                                                   Instructions for use
This is an example of reproducing the SwinIR brain MRI image super-resolution reconstruction task using MMEditing. Here's a brief user guide: First, the `datasets` folder contains several MRI DICOM folders. The `check` text file helps you convert DICOM files to usable PNG files. The `check` file also helps you process the original images to achieve 2x and 4x resolution. The `divide` file can split the dataset into test and training sets.

Regarding training, you can find the `train` file in the `tools` folder. If you need to modify parameters, you can find the configuration file in the `my config` folder. The trained model will be saved in the `work dirs` folder.

Regarding testing, you can find the `restoration demo` file in the `demo` folder. Simply modify the test dataset path and model path.

The following is the environment configuration:

conda create -n py38 python=3.8 -y

conda activate py38

cd code directory

pip install torch==1.10.0 torchvision==0.11.1 --index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard==2.11.0 timm segmentation-models-pytorch opencv-python einops yapf==0.40.1 setuptools==59.5.0 pytorch_msssim pytorch_wavelets PyWavelets scikit-image gradio==3.44.3