# Use Python 3.10 slim image as the base
FROM python:3.10-slim

# Set  working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install the requirements.txt
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    pip install -r requirements.txt

# Copy the app source code
COPY src /app/src

# Port accessible
EXPOSE 8008

# Run code
ENTRYPOINT ["streamlit", "run", "src.pipeline.frontend.app.py", "--server.port=8008", "--server.address=0.0.0.0"]
