# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV ACCEPT_EULA=Y \
    DEBIAN_FRONTEND=noninteractive

# ODBC + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 ca-certificates \
    gcc g++ make \
    unixodbc unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# MS ODBC 18 (official)
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy your analytics code
COPY analytics/ /app/

# Expose and run
ENV PORT=8000
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
