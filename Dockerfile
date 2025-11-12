# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV ACCEPT_EULA=Y \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps + ODBC
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 ca-certificates \
    unixodbc unixodbc-dev \
    gcc g++ make \
  && rm -rf /var/lib/apt/lists/*

# Microsoft ODBC Driver 18 for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the whole project (since repo root already has src/, models/, etc.)
COPY . /app/

# Expose and run
ENV PORT=8000
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
