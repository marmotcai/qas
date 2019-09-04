FROM python:3 AS env

MAINTAINER marmotcai "marmotcai@163.com"

RUN pip install --upgrade pip

ENV WORK_DIR=/root
WORKDIR $WORK_DIR

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

########################################################

FROM env AS server

ENV APP_DIR=$WORK_DIR/app
WORKDIR $APP_DIR

COPY . $APP_DIR

RUN pip install --no-cache-dir -r requirements.txt
RUN python manage.py makemigrations
RUN python manage.py migrate

ENV PORT=3080
RUN python manage.py manage.py runserver 0.0.0.0:$PORT
EXPOSE $PORT

########################################################

FROM env AS ssh

RUN sed -i '$a\alias ll=\"ls -alF\"' ~/.bashrc
RUN sed -i '$a\alias la=\"ls -A\"' ~/.bashrc
RUN sed -i '$a\alias l=\"ls -CF\"' ~/.bashrc

RUN apt-get update && \
    apt-get install -y wget vim openssh-server && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get autoremove -y && \
    apt-get clean

RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin yes/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config && \
    echo "root:112233" | chpasswd && \
    mkdir /var/run/sshd

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]

# ENV APP_PATH ${WORK_DIR}/app
# ENV PATH $PATH:$APP_PATH

# COPY . .
# CMD [ "python3", "./main.py", "-h" ]

