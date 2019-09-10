FROM marmotcai/pyrunner AS qas
MAINTAINER marmotcai "marmotcai@163.com"

ARG APP_GITURL="null"
ENV APP_PATH=${WORK_DIR}/app

ENV PIP_ALIYUN_URL="https://mirrors.aliyun.com/pypi/simple"
ENV PIP_TSINGHUA_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

RUN pip install --upgrade pip

RUN mkdir -p $APP_PATH
WORKDIR $APP_PATH

# COPY ./requirements.txt $APP_PATH/requirements.txt
COPY ./ ${APP_PATH}
RUN ls -la /root/app/*

# RUN pip install -i ${PIP_ALIYUN_URL} --no-cache-dir -r $APP_PATH/requirements.txt
RUN pip install --no-cache-dir -r $APP_PATH/requirements.txt

RUN python main.py -i "initcodefile"
RUN python manage.py makemigrations && \
    python manage.py migrate

RUN if [ "${APP_GITURL}" != "null" ] ; then \
      echo ${APP_GITURL} ;  \
      git clone ${APP_GITURL} ${APP_PATH}  ; \
      pip install --no-cache-dir -r $APP_PATH/requirements.txt ; \
    fi


COPY entrypoint.sh $WORK_DIR/entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 22 8000

# CMD ["/usr/sbin/sshd", "-D"]
# CMD ["$WORK_DIR/entrypoint.sh"]
CMD ["python", "/root/app/manage.py", "runserver", "0.0.0.0:8000"]
