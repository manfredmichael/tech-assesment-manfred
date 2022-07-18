import requests
import json
from PIL import Image

def transform_annotations(df):
    df['x1'] = df['left'] 
    df['x2'] = (df['left'] + df['width'])
    df['y1'] = df['top'] 
    df['y2'] = (df['top'] + df['height']) 

    df['top_left'] = df.apply(lambda x: list([x['x1'], x['y1']]), axis=1) 
    df['bottom_left'] = df.apply(lambda x: list([x['x1'], x['y2']]), axis=1) 
    df['bottom_right'] = df.apply(lambda x: list([x['x2'], x['y2']]), axis=1) 
    df['top_right'] = df.apply(lambda x: list([x['x2'], x['y1']]), axis=1) 
    annotations = df[['top_left', 'bottom_left', 'bottom_right', 'top_right']].values.tolist()
    return annotations 

def transform_scale(df, size):
    if len(df) > 0:
        x_ratio = 512/ size[0]
        y_ratio = 320 / size[1]

        df['left'] = df['left'] / x_ratio
        df['width'] = df['width'] / x_ratio
        df['top'] = df['top'] / y_ratio
        df['height'] = df['height'] / y_ratio

    return df


def inference(annotations):
    result = requests.post(
        f"http://localhost:5000/predict",
        files = {'file': open(f"img/annotation.jpeg", 'rb'),
                 'data': json.dumps({'annotations': annotations})
                },
    )

    return result.json()

