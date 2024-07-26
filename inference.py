import json
import os

import warnings

import SimpleITK
import torch

from mmseg.apis import inference_model, init_model


DOMAINS  = {'task1': 'body-part', 'task2': 'scanner'}

def read(path):
    image = SimpleITK.ReadImage(path)
    return SimpleITK.GetArrayFromImage(image)


def write(path, array):
    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(image, path, useCompression=True)


def main():
    task = 'task1'
    domain_path = f'/input/{DOMAINS[task]}.json'
    if os.path.exists(domain_path):
        domains = json.load(open(domain_path))
    else:
        domains = {}

    input_root = '/input/images/adenocarcinoma-image'
    output_root = '/output/images/adenocarcinoma-mask'

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    config = 'config.py'
    checkpoint = 'checkpoint.pth'
    model = init_model(config, checkpoint, device='cuda:0')
    
    with torch.no_grad():
        for filename in os.listdir(input_root):
            if filename.endswith('.mha'):
                output_path = f'{output_root}/{filename}'
                try:
                    input_path = input_root + '/' + filename
                    image = read(input_path)
                    warnings.warn(str(domains))
                    result = inference_model(model, image).pred_sem_seg.cpu().data
                    write(output_path, result.squeeze().numpy().astype('uint8'))
                except Exception as error:
                    print(error)


if __name__ == '__main__':
    main()