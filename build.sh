#!/bin/bash

cmd=${1}
app_path=${PWD}

case $cmd in
    build)
      docker build -t marmotcai/qas .
      exit 0
    ;;

    test)
      printf "test mode\n"
      docker run --rm -ti -v $app_path:/root/app  -p 9022:22 -p 8000:8000 marmotcai/qas /bin/bash
      exit 0
    ;;

    run)
      printf "app path: "$app_path"\n"
      docker rm -f my-qas
      docker run -v $app_path:/root/app -p 9022:22 -p 8000:8000 --name my-qas marmotcai/qas python /root/app/main.py -v
      exit 0
    ;;

    install)
      pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r requirements.txt
      exit 0
    ;;

  esac
    echo "use: sh build.sh build (docker: build image)"
    echo "use: sh build.sh test (docker: test image)"
    echo "use: sh build.sh run (docker: run image)"
    echo "use: sh build.sh install (python env install)"

exit 0;
