import os

import cv2
import torch

from mmseg.apis import inference_model, init_model

def main():
    input_root = '/input/images/adenocarcinoma-image'
    output_root = '/output/images/adenocarcinoma-mask'

    if not os.path.exists(output_root):
        os.mkdir(output_root)

    config = 'config.py'
    checkpoint = 'checkpoint.pth'
    model = init_model(config, checkpoint, device='cuda:0')
    with torch.no_grad():

        for filename in os.listdir(input_root):
            if filename.endswith('.png'):
                output_path = f'{output_root}/{filename}'
                try:
                    input_path = input_root + '/' + filename
                    result = inference_model(model, input_path).pred_sem_seg.cpu().data

                    cv2.imwrite(output_path, result.squeeze().numpy().astype('uint8'))
                except Exception as error:
                    print(error)


if __name__ == '__main__':
    main()