import mlflow.sagemaker as mfs

experiment_id = "0"
run_id = "96062ddd612a4c339a09af1c3abc31ed"
region = 'us-east-1'
aws_id = '048436951454'
arn = 'arn:aws:iam::048436951454:role/aws-sagemaker-for-deploy-ml-model' 
app_name = 'cac-model-application'
model_uri = f'mlruns/{experiment_id}/{run_id}/artifacts/model'
tag_id = '1.27.0'

image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:' + tag_id 

mfs.deploy(app_name,
           model_uri=model_uri,
           region_name=region,
           mode='create',
           instance_type='ml.m4.xlarge',
           execution_role_arn=arn,
           image_url=image_url,
           timeout_seconds=60*60)
