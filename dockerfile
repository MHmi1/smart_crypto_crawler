# Dockerfile

# Set the base image
FROM python:3.9.17-alpine

# Set the working directory
WORKDIR /app

# Confingure Python using environmental variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy the requirements file into the image and install them
COPY ./requirements.txt .
RUN python -m pip install --no-cache-dir -r ./requirements.txt

# Copy the source code into the image
COPY . .

# Expose the port
EXPOSE 8000

# Start the Uvicorn server
CMD ["python","./crawler.py"]