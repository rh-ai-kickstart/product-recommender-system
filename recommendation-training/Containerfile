FROM registry.access.redhat.com/ubi9/python-311

USER root
WORKDIR /app/

# install and activate env
COPY pyproject.toml pyproject.toml
RUN pip install uv
RUN uv sync
ENV VIRTUAL_ENV=.venv
ENV PATH=".venv/bin:$PATH"

COPY train-workflow.py train-workflow.py
COPY entrypoint.sh entrypoint.sh

# give premisssions and 
RUN chmod -R 777 . && ls -la
