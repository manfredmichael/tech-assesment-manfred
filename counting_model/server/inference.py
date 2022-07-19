from PIL import Image
import numpy as np
import json
from torchvision.transforms import transforms
import torch

from models import build_model
from config import cfg 
from loss import get_loss

cfg.merge_from_file('config/test_bmnet+.yaml')
device = torch.device(cfg.TRAIN.device)

criterion = get_loss(cfg)
criterion.to(device)
model = build_model(cfg, criterion, device)
model.to(device)


checkpoint = torch.load('checkpoints/bmnet+_pretrained/model_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])

def get_query_transforms(exemplar_size=(128, 128)):
    return transforms.Compose([
        transforms.Resize(exemplar_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_heatmap(outputs):
    std, mean = torch.std_mean(outputs)
    return transforms.Compose([
        transforms.Normalize(mean=mean.tolist(),
                             std=std.tolist())
    ])(outputs)

class MainTransform(object):
    def __init__(self):
        self.img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, img):
        img = self.img_trans(img)
        img = self.pad_to_constant(img, 32)
        return img 

    def pad_to_constant(self, inputs, psize):
        h, w = inputs.size()[-2:]
        ph, pw = (psize-h%psize),(psize-w%psize)
        # print(ph,pw)

        (pl, pr) = (pw//2, pw-pw//2) if pw != psize else (0, 0)   
        (pt, pb) = (ph//2, ph-ph//2) if ph != psize else (0, 0)
        if (ph!=psize) or (pw!=psize):
            tmp_pad = [pl, pr, pt, pb]
            # print(tmp_pad)
            inputs = torch.nn.functional.pad(inputs, tmp_pad)
        
        return inputs

def predict(img, annotations, return_density_map=False):
    w, h = img.size
    r = 1.0
    min_size = 384
    max_size = 1584
    box_number = 3
    scale_number = 20
    if h > max_size or w > max_size:
        r = max_size / max(h, w)
    if r * h < min_size or w*r < min_size:
        r = min_size / min(h, w)
    nh, nw = int(r*h), int(r*w)
    
    boxes = np.array(annotations, dtype='float32') * r   
    boxes = boxes[:box_number, :, :]

    query_transform = get_query_transforms()
    main_transform = MainTransform()

    patches = []
    scale_embedding = []

    for box in boxes:
        x1, y1 = box[0].astype(np.int32)
        x2, y2 = box[2].astype(np.int32)
        patch = img.crop((x1, y1, x2, y2))
        patches.append(query_transform(patch))
        # calculate scale
        scale = (x2 - x1) / nw * 0.5 + (y2 -y1) / nh * 0.5 
        scale = scale // (0.5 / scale_number)
        scale = scale if scale < scale_number - 1 else scale_number - 1
        scale_embedding.append(scale)


    img = main_transform(img)
    patches = torch.stack(patches, dim=0)
    patches = patches.view(1, 3, 3, 128, 128)
    scale_embedding = torch.tensor(scale_embedding).view(1, 3).long()
    patches = {'patches': patches,
               'scale_embedding': scale_embedding}



    img = img.to(device)
    patches['patches'] = patches['patches'].to(device)
    patches['scale_embedding'] = patches['scale_embedding'].to(device)

    model.eval()

    outputs = model(img, patches, is_train=False)
    heatmap = get_heatmap(outputs)
    heatmap = torch.squeeze(heatmap) 

    if return_density_map:
        return outputs.sum().tolist(), heatmap.tolist()
    return outputs.sum().tolist()

if __name__ == '__main__':
    filename = '1931.jpg'

    with open('data/annotation_FSC147_384.json') as f:
        annotations = json.load(f)

    img = Image.open(f'data/images_384_VarV2/{filename}').convert("RGB")
    annotations = annotations[filename]['box_examples_coordinates']

    print(predict(img, annotations))

