# Use official Python runtime as base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port (assuming the Flask app is running on port 5000)
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
