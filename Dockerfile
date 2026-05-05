# The dockerfile for creating an with all possible steps that
# can be run for the EEG project.

# Use official Python base image (use Python version from local machine)
FROM python:3.12

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY preprocessing/ /app
COPY misc/ /misc
COPY beam-python/ /beam-python

# Create a virtual environment in /opt/venv
RUN python -m venv /opt/venv

# Upgrade pip inside the virtual environment
RUN /opt/venv/bin/pip install --upgrade pip

# Install Python dependencies from requirements.txt
RUN /opt/venv/bin/pip install -r requirements.txt

# Ensure the virtual environment is used by default
ENV PATH="/opt/venv/bin:$PATH"

# Default command to run your application - we do not want to set one
# CMD ["python", "main.py"]

