# Official Algorithm Submission Example for the [Cosas Challenge](https://cosas.grand-challenge.org)

This example demonstrates an official algorithm submission implemented using MMSegmentation for the Cosas Challenge. The Cosas Challenge comprises two tasks, each involving datasets from different domains. During the preliminary test phase, each task's dataset is sourced from 4 domains. Each domain includes images of varying sizes, ranging from 1000 to 2000 pixels. In the final test phase, the datasets for each task are drawn from 6 domains, ranging from 1500 to 2000 pixels.

## Prerequisites

Before starting, ensure you have the following:

- Docker installed on your machine.
- Access to a GPU and configured Docker to use GPUs.

## Build

Clone the repository and build the Docker image with the following commands:

```bash
git clone https://github.com/VIBE-Lab/cosas-algorithm-submition.git
cd cosas-algorithm-submition
docker build -t cosas .
```

## Local Test
Assuming your local test dataset for task2 is in /cosas/task2/input/domain1, the folder structure should be as follows:
```
/cosas/task2/input/domain1
└──images/adenocarcinoma-image
    ├── image1.mha
    ├── image2.mha
    ...
    └── imagen.mha
```

Run the following command to test the algorithm locally:
```
sudo docker run --gpus all --volume /cosas/task2/input/domain1:/input --volume /cosas/task2/output:/output cosas
```

In the output image, regions with pixel values of 0 represent negative areas, while other regions indicate tumor areas. The output filename of the image will be in grayscale, png format, and the same size as the input image. The output folder structure will be as follows:
```
/cosas/task2/output
└──images/adenocarcinoma-mask
    ├── image1.mha
    ├── image2.mha
    ...
    └── imagen.mha
```

## Notes
- Ensure Docker has permission to access the specified directories.
- The path /cosas/task2/output should be an existing directory where the output will be saved. Create it if it does not exist.
- Adjust the paths in the Docker command according to your local setup if they differ.
Verify that all images in the input folder are in the .mha format

If you have any questions, please feel free to contact the challenge organizer via email:

- Official email: vibe.research@outlook.com
- Xi Long: xi.loong@outlook.com
- Biweng Meng: Biwen.Meng22@student.xjtlu.edu.cn
