FROM continuumio/miniconda3 AS builder

RUN pip install git+https://github.com/conda/conda-pack@master

COPY deploy/conda_envs/linux-cpu-64-dev.lock env.lock
RUN conda create --name pyenv --file env.lock

RUN /bin/bash -c "conda-pack -n pyenv -o /tmp/env.tar && mkdir venv && cd venv && tar xf /tmp/env.tar && rm /tmp/env.tar"

RUN venv/bin/conda-unpack

FROM debian:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 APP_USER=app APP_HOME=/home/app

RUN useradd --no-log-init -r -m -U "$APP_USER"

COPY --from=builder --chown="$APP_USER":"$APP_USER" venv "$APP_HOME"/pyenv
COPY --chown="$APP_USER":"$APP_USER" ./ "$APP_HOME"/app

USER "$APP_USER"
WORKDIR "$APP_HOME"/app

RUN if [ -f "$APP_HOME/app/deploy/conda_envs/requirements-linux-cpu-64-dev.txt" ] ; then \
   /bin/bash -c "$APP_HOME/pyenv/bin/python -m pip install -r $APP_HOME/app/deploy/conda_envs/requirements-linux-cpu-64-dev.txt --no-deps"; \
  else \
    echo "File not Found"; \
  fi

RUN /bin/bash -c "$APP_HOME/pyenv/bin/python -m pip install -e $APP_HOME/app"

ENV PATH="$APP_HOME/pyenv/bin:$PATH"

RUN mkdir data artifacts logs

ENTRYPOINT ["python", "production/cli.py"]
CMD ["job", "list"]
