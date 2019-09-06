FROM marmotcai/pyrunner AS qas
MAINTAINER marmotcai "marmotcai@163.com"

ARG APP_GITURL="NULL"
ENV PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"

WORKDIR ${WORK_DIR}
COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install  -i ${PIP_INDEX_URL} --no-cache-dir -r ./requirements.txt

ENV APP_PATH=${WORK_DIR}/app/
RUN if [ "${APP_GITURL}" != "NULL" ] ; then echo ${APP_GITURL} ;  git clone ${APP_GITURL} ${APP_DIR} ; fi
WORKDIR APP_PATH

COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh
EXPOSE 22 8000

# CMD ["/usr/sbin/sshd", "-D"]
# CMD ["ls"]
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
