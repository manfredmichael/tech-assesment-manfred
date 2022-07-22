# BMNet+ Deployment with Docker and Flask

Try it your self: https://manfredmichael-streamlit-interface-app-vn2ph1.streamlitapp.com/

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/counting-model-demo.gif?raw=true)

## Deployment Stack

I created a Flask app for model inference. Containerized the app using docker and deployed it to AWS EC2 on a t3.medium instance. Then, I developed the streamlit interface to demonstrate model prediction & interact with image ROI.

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/pipeline.png?raw=true)

## Deployment Instructions
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

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/2.1.png?raw=true)

For **Instance type**, change the default choice to **t3.medium**.

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/2.2.png?raw=true)

For **Key pair**, you could create new key pair if you don't have one. When you create a new one, a `.pem` file would automatically be downloaded into your device.

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/2.3.png?raw=true)

For **Network settings**, click on Edit. Then, click on **Add security group rule**. For the new rule, set the **Type** to HTTP, and the **Source Type** to Anywhere.

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/2.4.png?raw=true)

For **Configure storage**, insert at least **12 GiB**. 

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/2.5.png?raw=true)

Then click on **Launch Instance** to launch this new instance.

3. Connecting to your EC2 Instance.

First, select your newly created instance, then click **Action** and choose **Connect**.

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/3.1.png?raw=true)

You will find the instructions on how to connect to your instance through your local terminal.

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/3.2.png?raw=true)

Open your local terminal, and follow the instructions showed. Then, you will be connected to the instance.

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/3.3.png?raw=true)

4. Deploying the dockerized model on EC2 Instance.

First, update the instance

```sudo apt-get update```

Then, install docker-compose and awscli

```
sudo apt install docker-compose
sudo apt install awscli
```

Then, configure your awscli. Insert your Acess Key & Secret Access Key, and region. For the **output format** insert **json**. You could get your Access Key from your IAM role. Make sure your IAM role has **S3 full access** permission.

```aws configure```

Next, clone this repository to the EC2 Instance.

```git clone https://github.com/manfredmichael/tech-assesment-manfred.git```

Go to the `counting_model` directory

```cd tech-assesment-manfred/counting_model```

Get the model weights from your S3 Bucket

```aws s3 sync s3://counting-model-bucket/checkpoints checkpoints```

The `checkpoints/` folder should appear inside the `server/` directory.

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/4.1.png?raw=true)

Before we start building the docker, create a docker group and add your user.

```
sudo groupadd docker
sudo usermod -aG docker ${USER}
```

Then, build the docker (Note: make sure you are in `tech-assesment-manfred/counting-model/` directory).

```
docker build -t counting_model .
```

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/4.2.png?raw=true)

Now we can run the docker image

```
docker run --rm -p 80:80 counting_model
```

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/4.3.png?raw=true)

On a different terminal, now you can send a request to the flask API through your Instance public DNS.

![](https://github.com/manfredmichael/tech-assesment-mlflow-amazon-sagemaker/blob/main/assets/4.3.png?raw=true)


