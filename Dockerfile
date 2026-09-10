FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Render provides the runtime port through the PORT environment variable.
EXPOSE 10000

# Start the Flask app
CMD ["sh", "-c", "flask run --host=0.0.0.0 --port=${PORT:-7860}"]