# Use an official lightweight Python image: https://hub.docker.com/_/python
FROM python:3.10-slim as python-base

# Set environment variables:
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_VERSION=1.8
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
# Poetry virtual environment cache 
ENV POETRY_CACHE_DIR=/opt/.cache


# Create stage for Poetry installation
FROM python-base as poetry-base

# Creating a virtual environment just for poetry and install it with pip
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}


# Create a new stage from the base python image
FROM python-base as bandim-fastapi

# Copy Poetry to app image
COPY --from=poetry-base ${POETRY_VENV} ${POETRY_VENV}

# Add Poetry to the PATH variable
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Set the working directory in the container
WORKDIR /app

# Copy dependencies
COPY poetry.lock pyproject.toml ./

# Install poetry and all dependencies
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry install --no-interaction --no-cache

# Copy the current directory contents into the container at /code
COPY ./bandim-api /app

# Command to run the uvicorn server
CMD ["poetry", "run", "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]