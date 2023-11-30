#!/bin/bash
VOLUME_NAME="nlp_tp2_volume"
NAME="nlp_tp2"

git pull
docker build -t $NAME:latest .

# Unmounting and clearing volumes from the main image
CONTAINERS=$(docker ps -a --filter volume=$VOLUME_NAME -q)

for CONTAINER_ID in $CONTAINERS; do
  echo "Unmounting volume $VOLUME_NAME from container $CONTAINER_ID"
  docker rm -v $CONTAINER_ID
done

echo "Volume $VOLUME_NAME unmounted from all containers!"
docker volume rm $VOLUME_NAME
echo "Volume $VOLUME_NAME removed!"

docker volume create $VOLUME_NAME
echo -e "\n"
docker run -v $VOLUME_NAME:/app $NAME:latest

# Copying volume files to disk
if [ -d "models" ]; then
  # The directory exists, do nothing
  :
else
    mkdir models
fi

CONTAINER_ID=$(docker ps -a --filter ancestor=$NAME -q)
rm models/*
docker cp $CONTAINER_ID:/app/models/. models