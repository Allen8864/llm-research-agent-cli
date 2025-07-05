FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY ./src /app/src

# Set the entrypoint for the container
# This allows running the agent with `docker run agent "<question>"`
ENTRYPOINT ["python", "-m", "src.agent.main"]
