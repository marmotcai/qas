FROM marmotcai/pyrunner AS qas
MAINTAINER marmotcai "marmotcai@163.com"

ENV APP_PATH=${WORK_DIR}/app/

WORKDIR ${WORK_DIR}

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]

# CMD ["ls"]
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
