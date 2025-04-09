# Use a Python 3.9 slim image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (excluding the data folder because of .dockerignore)
COPY . .

# Expose the port that Dash will run on
EXPOSE 8050

# Command to run the Dash app
CMD ["python", "data_analysis.py"]
