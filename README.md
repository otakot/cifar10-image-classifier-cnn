## Build the docker image from project sources

    docker build -t image-classifier --platform=linux/amd64 .

## Run docker image

    docker run --platform linux/amd64 --rm -p 80:80 -it image-classifier

Note!:  --platform linux/amd64 is  needed only if commands are excute in non liunux environment (e.g MacOS)

