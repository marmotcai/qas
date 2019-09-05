#!/usr/bin/env bash

app_path=$PWD

if [[ 'MINGW' =~ $uname ]]; then
  app_path="/"$app_path
fi

docker build -t marmotcai/qas . 

docker rm -f my-qas


cmd=${1}
if [[ $cmd =~ 'test' ]]; then
  printf "test mode\n"
  docker run --rm -v $app_path:/root/app -p 9022:22 -p 8000:8000 marmotcai/qas /bin/bash
fi

printf "app path: "$app_path"\n"

docker run -d -ti --name my-qas -v $app_path:/root/qas -p 9022:22 -p 8000:8000 marmotcai/qas
winpty docker exec -it my-qas //bin/bash
