FROM ubuntu:22.04

# Install necessary dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files to the working directory
COPY __init__.py app.py preprocessing.py tracker.py requirements.sh config.py /app/

RUN mkdir /app/models
COPY ./models/* /app/models/

RUN mkdir /app/data
COPY ./data/* /app/data/

RUN mkdir /app/Results

RUN pip3 install --upgrade pip
# Install Python dependencies
RUN sh ./requirements.sh

# Expose the FastAPI port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["/usr/local/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]