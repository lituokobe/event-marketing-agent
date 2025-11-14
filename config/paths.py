from pathlib import Path

# Get project folder dir
current_file = Path(__file__).resolve()
project_dir = current_file.parent.parent

# Data paths
AGENT_DATA_PATH = project_dir / "data/agent_data.json"
INTENTION_PATH = project_dir / "data/intentions.json"
KNOWLEDGE_PATH = project_dir / "data/knowledge.json"
CHATFLOW_DESIGN_PATH = project_dir / "data/chatflow_design.json"
DIALOG_PATH = project_dir / "data/dialogs.json"
VECTOR_BD_COLLECTION_PATH = project_dir / "data/vector_db_collection.json"

# Other paths
ENV_PATH = project_dir / '.env'
VECTOR_DB_PATH = project_dir / "chroma_db"
LOG_PATH = project_dir / "logs"