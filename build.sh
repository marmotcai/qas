#!/usr/bin/env bash

docker build -t marmotcai/qas .

cmd=${1}
if [[ $cmd =~ 'test' ]]; then
  printf "test mode\n"
  docker run --rm -v $app_path:/root/app -p 9022:22 -p 8000:8000 marmotcai/qas python /root/app/manage.py runserver 0.0.0.0:8000
fi

printf "app path: "$app_path"\n"
docker rm -f my-qas
docker run -v $app_path:/root/app -p 9022:22 -p 8000:8000 --name my-qas marmotcai/qas python /root/app/main.py -v
