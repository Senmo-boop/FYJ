Instructions for Use

This is an example of reproducing the SwinIR brain MRI image super-resolution reconstruction task using MMEditing. Here is a brief user guide: Regarding the data preprocessing script `preprocess.py`, you can directly run the following command in the command line to obtain a ready-to-use dataset:

python "G:\code\project\code\preprocess.py" --source G:\fastMRI_brain_DICOM --output G:\datasets --train 100 --val 10 --test 10 --seed 42

All packages required for preprocessing scripts:

pip install pydicom opencv-python numpy tqdm

Regarding training, you can find the `train` file in the `tools` folder. You can directly run it in the command line using swinir2x as an example:

python "G:\code\project\code\tools\train.py" --config G:\code\project\code\my_config_2x\swinir2x.py --work-dir G:\code\project\code\work_dirs\swinir_2x

Regarding testing, you can find the `test` file in the `tools` folder. You can run the following command directly in the command line, using swinir2x as an example:

python "G:\code\project\code\tools\test.py" --config G:\code\project\code\my_config_2x\swinir2x.py --checkpoint G:\code\project\code\work_dirs_2x\swinir2x\iter_20000.pth --save-path G:\code\project\code\test_results\swinir_2x

Regarding testing the model's image super-resolution reconstruction: you can use the following command: 
python G:\code\project\code\demo\restoration_demo.py --config G:\code\project\code\work_dirs_2x\swinir2x\swinir2x.py --checkpoint G:\code\project\code\work_dirs_2x\swinir2x\iter_20000.pth --img_path_dir` G:\code\project\code\datasets\test\LRx2 --save_path_dir G:\code\project\code\datasets\pred\swinir2x

The following is the environment configuration:

conda create -n py38 python=3.8 -y

conda activate py38

cd code directory

pip install torch==1.10.0 torchvision==0.11.1 --index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard==2.11.0 timm segmentation-models-pytorch opencv-python einops yapf==0.40.1 setuptools==59.5.0 pytorch_msssim pytorch_wavelets PyWavelets scikit-image gradio==3.44.3




