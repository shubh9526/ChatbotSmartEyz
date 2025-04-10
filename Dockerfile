# Base Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y

# Install system dependencies and Microsoft ODBC driver
RUN apt-get update && apt-get install -y \
    apt-utils \
    curl gnupg apt-transport-https \
    unixodbc unixodbc-dev \
    gcc g++ \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl -sSL https://packages.microsoft.com/config/debian/10/prod.list -o /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update && apt-get install -y msodbcsql17 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Expose port (update if needed)
EXPOSE 8000

# Default command to run your app
CMD ["python", "app.py"]
