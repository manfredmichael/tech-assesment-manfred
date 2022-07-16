import requests
import json
import argparse
from PIL import Image


parser = argparse.ArgumentParser(
    description="CAC Model inference test"
)
parser.add_argument(
    "--img",
    default="1391.jpg",
    metavar="FILE",
    help="image filename",
    type=str,
)

args = parser.parse_args()
filename = args.img

with open('data/annotation_FSC147_384.json') as f:
    annotations = json.load(f)

annotations = annotations[filename]['box_examples_coordinates']

result = requests.post(
    f"http://127.0.0.1:5000/predicts",
    files = {'file': open(f"data/images_384_VarV2/{filename}", 'rb'),
             'data': json.dumps({'annotations': annotations}) 
            },
)

print(result.json())
# ).json()


