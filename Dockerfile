FROM python:3 AS env
MAINTAINER marmotcai "marmotcai@163.com"

########################################################

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

CMD ["/usr/sbin/sshd", "-D"]

########################################################

RUN pip install --upgrade pip

ENV WORK_DIR=/root
WORKDIR $WORK_DIR

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 22 8000

########################################################

FROM env AS runer

ENV APP_GIT_URL="https://marmotcai:aa!112233@github.com/marmotcai/qas.git"
ENV APP_PATH=${WORK_DIR}/app/

WORKDIR $APP_PATH
RUN git clone $APP_GIT_URL $APP_PATH

CMD ["python", "./main.py", "-u"]
