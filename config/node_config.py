from config.config_setup import NodeConfig, AgentConfig
from config.paths import KEYWORD_JSON_PATH, SEMANTIC_JSON_PATH, LLM_JSON_PATH, DB_PATH, DB_EMBEDDING_MODEL_NAME

# Set once at startup
AgentConfig.set_global(
    AgentConfig(
        keyword_json_path=str(KEYWORD_JSON_PATH),
        semantic_json_path=str(SEMANTIC_JSON_PATH),
        llm_json_path=str(LLM_JSON_PATH),
        db_path=str(DB_PATH),
        db_embedding_model_name=DB_EMBEDDING_MODEL_NAME
    )
)

# Create node configs without repeating paths
general_config = NodeConfig(
    default_reply="这是一条默认回答。",
    db_collection_name="general_collection",
    db_collection_metadata = {"hnsw:space": "cosine"},
    llm_name="qwen_llm",
    cosine_threshold=0.82,
    active_ids=["001", "002", "004", "008"],
    enable_logging=True
)

# config_8 = NodeConfig(
#     active_ids=["008"],
#     db_collection_name="collection_8",
#     db_collection_metadata = {"hnsw:space": "cosine"},
#     llm_name="qwen_llm",
#     cosine_threshold=0.85,
#     enable_logging=True
# )