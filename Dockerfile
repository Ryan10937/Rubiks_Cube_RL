# Use official Python image as base
FROM python:3.12.3-slim

# Set working directory
WORKDIR /app
RUN apt-get update && apt-get install -y python3-tk && rm -rf /var/lib/apt/lists/*

# Copy requirements if available, else skip
COPY install/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
# Copy only folders containing .py files
COPY scripts/*.py scripts/
COPY app.py app.py
COPY rubiks_utils/*.py rubiks_utils/
COPY pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir -e .

# Copy the structure of the history folder
RUN mkdir history
RUN mkdir history/general_history


RUN pip freeze
ENV UPLOAD_IP=$UPLOAD_IP
ENV NUM_EPISODES=$NUM_EPISODES
ENV NUM_TIMESTEPS=$NUM_TIMESTEPS
# Set default command to run the script
# CMD ["sh","-c","python3", "scripts/main.py --generate_data --num_episodes $NUM_EPISODES --num_timesteps $NUM_TIMESTEPS --upload_ip $UPLOAD_IP"]

ENTRYPOINT ["echo Hello from Docker!"]