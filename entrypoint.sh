#!/bin/bash

cmd=${1}
param=${2}

case $cmd in 
    init)
      pip install --no-cache-dir -r requirements.txt      

      python main.py -i "initcodefile"
      python manage.py makemigrations
      python manage.py migrate
      exit 0
    ;;

    run)
      python main.py -s start
      python manage.py runserver 0.0.0.0:8000
      exit 0
    ;;

    daemon)
      python main.py -d
      exit 0
    ;;

esac
    echo "use: ./entrypoint.sh init"
    echo "use: ./entrypoint.sh run"
    echo "use: ./entrypoint.sh daemon"

exit 0
