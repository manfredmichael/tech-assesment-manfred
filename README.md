# BMNet+ Deployment with Docker and Flask

Try it your self: https://manfredmichael-streamlit-interface-app-vn2ph1.streamlitapp.com/

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/counting-model-demo.gif?raw=true)

## Deployment Stack

I created a Flask app for model inference. Containerized the app using docker and deployed it to AWS EC2 on a t3.medium instance. Then, I developed the streamlit interface to demonstrate model prediction & interact with image ROI.

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/pipeline.png?raw=true)

## Deploying Instructions
### Deploying Model Inference on Amazon EC2
1. Uploading model weights to S3 Bucket

Download the [bmnet+_pretained](https://www.dropbox.com/s/mr52q8kp9tp7cy9/model_best.pth?dl=0) & [resnet50](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar) model. Then, create a new bucket called `counting-model-bucket` and upload the model weights in this directory structure
```
checkpoints/
├─ bmnet+_pretrained/
│  ├─ model_best.pth
├─ resnet50/
│  ├─ swav_800ep_pretrain.pth.tar
```

2. Create New EC2 Instance

Create a New EC2 Instance with these configurations:

For **Application and OS Images (Amazon Machine Image)**, choose **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**.

For **Instance type**, change the default choice to **t3.medium**.

For **Key pair**, you could create new key pair if you don't have one. When you create a new one, a `.pem` file would automatically be downloaded into your device.

For **Network settings**, click on Edit. Then, click on **Add security group rule**. For the new rule, set the **Type** to HTTP, and the **Source Type** to Anywhere.

For **Configure storage**, insert at least **12 GiB**. 

Then click on **Launch Instance** to launch this new instance.

