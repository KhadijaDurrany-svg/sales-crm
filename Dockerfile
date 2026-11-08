# Minimal, offline-friendly image. No network is needed at runtime.
FROM python:3.12-slim

# Keep Python lean and predictable in a container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# Create and switch to a non-root user.
RUN useradd --create-home --uid 1001 leadsift
USER leadsift

ENTRYPOINT ["leadsift"]
CMD ["bench", "--no-write"]
