This is simple python application that builds and trains small CNN for classification of images from CIFAR10 dataset. Its based on official example from pytoch framework decumentaiton: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html. Application is supposed to be built, deployed and executed as docker image.

## Build the docker image from project sources

    docker build -t image-classifier --platform=linux/amd64 .

## Run docker image

    docker run --platform linux/amd64 --rm -p 80:80 -it image-classifier

Note!:  --platform linux/amd64 is  needed only if commands are excute in non liunux environment (e.g MacOS)

