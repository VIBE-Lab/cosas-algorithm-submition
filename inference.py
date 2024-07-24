import os

import cv2
import torch

from mmseg.apis import inference_model, init_model

def main():
    input_root = '/input'
    output_root = '/output'
    config = 'config.py'
    checkpoint = 'checkpoint.pth'
    model = init_model(config, checkpoint, device='cuda:0')
    with torch.no_grad():
        for domain in os.listdir(input_root):
            input_dir = f'{input_root}/{domain}/image'
            output_dir = f'{output_root}/{domain}/image'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            for filename in os.listdir(input_dir):
                if filename.endswith('.png'):
                    output_path = f'{output_dir}/{filename}'
                    try:
                        input_path = input_dir + '/' + filename
                        result = inference_model(model, input_path).pred_sem_seg.cpu().data

                        cv2.imwrite(output_path)
                    except Exception as error:
                        print(error)



if __name__ == '__main__':
    main()