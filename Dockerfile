FROM marmotcai/pyrunner AS qas
MAINTAINER marmotcai "marmotcai@163.com"

ENV APP_PATH=${WORK_DIR}/app/

WORKDIR ${WORK_DIR}
CMD ["ls"]
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
