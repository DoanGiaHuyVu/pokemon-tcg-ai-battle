FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
RUN pip install torch numpy

# Copy all playground files
COPY . .

# Default command
CMD ["python3", "local_runner.py"]
