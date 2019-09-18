#!/bin/bash

imagename="marmotcai/qas"
cmd=${1}
param=${2}
app_path=${PWD}

case $cmd in
    build)
      if [[ $param =~ 'git' ]]; then
        docker build --build-arg APP_GITURL=$param -t ${imagename} .
      else
	docker build -t ${imagename} .
      fi
      exit 0
    ;;

    test)
      printf "test mode\n"
      docker run --rm -ti -p 3280:8000 -p 3222:22 -v $PWD:/root/qas ${imagename} /bin/bash
      exit 0
    ;;

    debug)
      printf "app path: "$app_path"\n"
      docker rm -f my-qas
      # docker run -p 3222:22 -p 3280:8000 -v /home/x/myfiles/workspace/qas:/root/app --name my-qas marmotcai/qas /bin/bash
      docker run -ti -d -p 3280:8000 -p 3222:22 --name my-qas -v $PWD:/root/qas ${imagename} # python /root/app/manage.py runserver 0.0.0.0:8000
      exit 0
    ;;

    run)
      cpwd=`pwd`

      # docker rm -f my-smb
      # docker run -ti -d --privileged=true \
      #         --name my-smb \
      #         --publish 445:445 \
      #         --publish 137:137 \
      #         --publish 138:138 \
      #         --publish 139:139 \
      #          --volume ${cpwd}:/share \
      #         --env workgroup=${2:-workgroup} \
      #         marmotcai/smb

      docker rm -f my-qas
      docker run -ti -d \
                 -p 3280:8000 -p 3222:22 \
                 --name my-qas \
                 --volume ${cpwd}:/root/app \
                 ${imagename} # python /root/app/manage.py runserver 0.0.0.0:8000
      exit 0
    ;;

    install)
      pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r requirements.txt
      exit 0
    ;;

  esac
    echo "use: sh build.sh build"
    echo "use: sh build.sh build https://github.com/marmotcai/qas.git"
    echo "use: sh build.sh test"
    echo "use: sh build.sh debug"
    echo "use: sh build.sh run"
    echo "use: sh build.sh install"

exit 0;
