# BMNet+ Deployment with Docker and Flask

## Preparation

Download the pretrained model [here](https://www.dropbox.com/s/mr52q8kp9tp7cy9/model_best.pth?dl=0). Then, please follow the [FSC-147 official repository](https://github.com/cvlab-stonybrook/LearningToCountEverything) to download and unzip the dataset. Then, please place the data lists  ``data_list/train.txt``, ``data_list/val.txt`` and ``data_list/test.txt`` in the dataset directory. Note that, you should also download data annotation file ``annotation_FSC147_384.json`` and ``ImageClasses_FSC147.txt`` file from [Link](https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master/data) and place them in the dataset folder. Final the path structure used in our code will be like :
````
$PATH_TO_DATASET/
├──── gt_density_map_adaptive_384_VarV2
│    ├──── 6146 density maps (.npy files)
│    
├──── images_384_VarV2
│    ├──── 6146 images (.jpg)
│ 
├────annotation_FSC147_384.json (annotation file)
├────ImageClasses_FSC147.txt (category for each image)
├────Train_Test_Val_FSC_147.json (official data splitation file, which is not used in our code)
├────train.txt (We generate the list from official json file)
├────val.txt
├────test.txt

````

## Tracking the training with mlflow

After running the command below, an artifact of the model will appear on `mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model`
```
python3 train.py --cfg "config/bmnet+_fsc147.yaml"
```

## Build a Docker Image and push it to AWS ECR

Pick one of the model artifacts and build the docker image.
```
cd 'mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model'
mlflow sagemaker build-and-push-container
```

