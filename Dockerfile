FROM python:3.11-slim

COPY ./requirements.txt /tmp/requirements.txt
COPY ./entrypoint.sh /tmp/entrypoint.sh
COPY ./src /app/src
COPY ./data /app/data

RUN chmod +x /tmp/entrypoint.sh

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install gcc and curl for building and downloading packages
# Use --no-install-recommends to avoid installing unnecessary packages
# Clean up apt cache to reduce image size
# Use pip to install Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl && \
    pip install --upgrade pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

ENV PYTHONPATH=/app

# Port to expose
EXPOSE 33000

# Health Check
# A call to http://localhost:33000/check_health will be made. The api returns 0 if the database is reachable, 1 otherwise.
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:33000/check_health || exit 1

# Create a non-root user 'appuser'
RUN useradd --create-home appuser
# add write permissions to the /app/data directory
RUN chown -R appuser:appuser /app/data
# switch to this user
USER appuser


# CMD with JSON notation
CMD ["/tmp/entrypoint.sh"]