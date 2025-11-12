# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV ACCEPT_EULA=Y \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Base system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg ca-certificates \
    unixodbc unixodbc-dev \
    gcc g++ make \
  && rm -rf /var/lib/apt/lists/*

# --- Microsoft ODBC Driver 18 for SQL Server (Debian 12 / keyring method) ---
# Add Microsoft signing key to keyring and repo list (no apt-key)
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
    > /etc/apt/sources.list.d/microsoft-prod.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the whole project (your repo root already has src/, models/, etc.)
COPY . /app/

# Expose and run
ENV PORT=8000
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
