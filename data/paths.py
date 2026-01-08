from pathlib import Path

# Get project folder dir
current_file = Path(__file__).resolve()
project_dir = current_file.parent.parent

ENV_PATH = project_dir / ".env"
LOG_PATH = project_dir / "logs"

# Embedding service
# EMBED_SERVICE_URL = "http://192.168.0.143:8081" # Tuo local deployment, port 8081
EMBED_SERVICE_URL = "http://192.168.0.143:8081" # also deployed on the server, port 8083