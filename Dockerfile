# Use an official PyTorch runtime as a parent image
# Note! not needed if docker image will deployed/executed in VM instance with already available pytorch runtime
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set the working directory to /image-classifier
WORKDIR /image-classifier

# Copy project sources into the container at /image-classifier
ADD ./src /image-classifier
ADD ./requirements.txt /image-classifier
ADD ./data /image-classifier/data

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run main.py when the container launches
CMD ["python", "main.py"]