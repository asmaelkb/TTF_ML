# Dockerfile

# pull the official docker image
FROM python:3.11.1-slim

# set work directory
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV MLFLOW_TRACKING_URI=http://172.16.239.131:5000

# Run mlflow server when the container launches
CMD ["mlflow", "server", "--host", "0.0.0.0"]