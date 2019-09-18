FROM marmotcai/pyrunner AS qas
MAINTAINER marmotcai "marmotcai@163.com"

ARG APP_GITURL="null"
ENV APP_PATH=${WORK_DIR}/app

ENV PIP_ALIYUN_URL="https://mirrors.aliyun.com/pypi/simple"
ENV PIP_TSINGHUA_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
# RUN timedatectl status

RUN apt update && \
    apt install -y graphviz && \
    apt autoremove && \
    apt clean

RUN pip install --upgrade pip

RUN mkdir -p $APP_PATH
RUN if [ "${APP_GITURL}" != "null" ] ; then \
      echo ${APP_GITURL} ;  \
      git clone ${APP_GITURL} ${APP_PATH}  ; \
      pip install --no-cache-dir -r $APP_PATH/requirements.txt ; \
    fi

COPY ./requirements.txt $APP_PATH/requirements.txt

RUN pip install -i ${PIP_TSINGHUA_URL} --no-cache-dir -r $APP_PATH/requirements.txt # RUN pip install --no-cache-dir -r ./requirements.txt

# COPY ./ ${APP_PATH}
# RUN ls -l $APP_PATH/entrypoint.sh
# RUN chmod +x $APP_PATH/entrypoint.sh # \ sh $APP_PATH/entrypoint.sh init

WORKDIR $APP_PATH

EXPOSE 22 8000
VOLUME $APP_PATH

CMD ["/usr/sbin/sshd", "-D"] # CMD ["./entrypoint.sh", "run"]
