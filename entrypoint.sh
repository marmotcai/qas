#!/bin/bash

cd $APP_PATH
python main.py -i "initcodefile"
python manage.py makemigrations
python manage.py migrate
python manage.py runserver 0.0.0.0:8000

# python main.py -m 300096

