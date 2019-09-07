FROM marmotcai/pyrunner AS qas
MAINTAINER marmotcai "marmotcai@163.com"

ARG APP_GITURL="null"
ENV APP_PATH=${WORK_DIR}/app

ENV PIP_ALIYUN_URL="https://mirrors.aliyun.com/pypi/simple"
ENV PIP_TSINGHUA_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

RUN pip install --upgrade pip

RUN mkdir -p ${APP_PATH}
COPY requirements.txt ${APP_PATH}/requirements.txt

RUN if [ "${APP_GITURL}" != "null" ] ; then \
      RUN echo ${APP_GITURL} ;  \
      git clone ${APP_GITURL} ${APP_PATH}  ; \
    fi

RUN pip install  -i ${PIP_ALIYUN_URL} --no-cache-dir -r $APP_PATH/requirements.txt

COPY entrypoint.sh $WORK_DIR/entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 22 8000

CMD ["/usr/sbin/sshd", "-D"]
# CMD ["$WORK_DIR/entrypoint.sh"]
