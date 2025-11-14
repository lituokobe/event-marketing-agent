import json
from pathlib import Path
from typing import Literal, Any
from pydantic import BaseModel, Field

# Configuration at agent level, the params won't change across nodes
class AgentConfig(BaseModel):
    intention_priority: Literal["知识库优先", "回答分支优先", "智能匹配优先"] = Field(..., description="Priority to infer intentions")
    use_llm: bool = Field(..., description="Use LLM for semantic matching")
    llm_name: str = Field(..., description="LLM instance name")
    llm_threshold: int = Field(..., description="The threshold count of user input to use LLM")
    llm_context_rounds: int = Field(..., description="The rounds of chat history for LLM to use")
    llm_role_description: str = Field(..., description="The description of the LLM role")
    llm_background_info: str = Field(..., description="The background information for the LLM role")
    cosine_threshold: float = Field(..., description="Semantic match threshold")
    vector_db_path: str = Field(..., description="Local path for the vector DB")

# class that holds information related to knowledge base, e.g. data, mapping, matchers.
class KnowledgeContext(BaseModel):
    knowledge: list = Field(..., description="Knowledge data")
    infer_id: dict = Field(..., description="Mapping: knowledge name -> ID")
    infer_description: dict = Field(..., description="Mapping: knowledge name -> description")
    type_lookup: dict = Field(..., description="Mapping: knowledge ID -> knowledge type")
    keyword_matcher: Any = Field(None, description="Knowledge Keyword Matcher instance (if initialized)")
    semantic_matcher: Any = Field(None, description="Knowledge Semantic Matcher instance (if initialized)")

# Configuration at node level
class NodeConfig(BaseModel):
    # Flow level fields
    main_flow_id: str|None = Field(None, description="Main flow ID, unique ID specified by system.")
    main_flow_name: str|None = Field(None, description="Main flow name, created by user.")
    # Node-specific fields
    node_id: str|None = Field(None, description="Base node ID, unique ID specified by system.")
    node_name: str|None = Field(None, description="Base node name, created by user.")
    default_reply_id_list: list|None = Field(None, description="Default reply ID of the node when the workflow is directed here.")
    db_collection_name: str|None = Field(None, description="Vector DB collection name")
    intention_branches: list[dict|None]|None = Field(None, description="branches of intentions that includes intention IDs")
    transfer_node_id: str|None = Field(None, description="Action to execute for transfer node")
    enable_logging: bool = Field(False, description="Enable debug logs")
    # Global config fields at agent level
    agent_config: AgentConfig = Field(..., description="Agent-level configuration")

class ChatFlowConfig(BaseModel):
    agent_config:AgentConfig|None
    knowledge_context:KnowledgeContext|None
    chatflow_design:dict
    intentions:list[dict|None]
    dialog_lookup:dict[str, dict]
    vector_db_collection_lookup:dict[str, str]

    @classmethod
    def from_paths(
            cls,
            agent_data_path: str | Path,
            knowledge_path: str | Path,
            chatflow_design_path: str | Path,
            intention_path: str | Path,
            dialog_path: str | Path,
            vector_db_collection_path: str | Path,
            vector_db_path: str | Path
    )->"ChatFlowConfig":
        # Load agent data
        with open(agent_data_path, 'r', encoding='utf-8') as f:
            agent_data = json.load(f)

        # Initialize agent level data configuration with the loaded agent data
        agent_config = AgentConfig(
            intention_priority = str(agent_data["intention_priority"]),
            use_llm=bool(agent_data["use_llm"]),
            llm_name=str(agent_data["llm_name"]),
            llm_threshold= int(agent_data["llm_threshold"]),
            llm_context_rounds= int(agent_data["llm_context_rounds"]),
            llm_role_description= str(agent_data["llm_role_description"]),
            llm_background_info= str(agent_data["llm_background_info"]),
            cosine_threshold=float(agent_data["cosine_threshold"]),
            vector_db_path=str(vector_db_path),
        )

        # Load knowledge and prepare knowledge context
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
        knowledge_infer_id = {}  # dict to store knowledge type_name -> type_id
        knowledge_infer_description = {}  # dict to store knowledge type_name -> description
        knowledge_type_lookup = {}  # dict to store knowledge type_id -> knowledge_type
        if knowledge:
            for item in knowledge:
                knowledge_infer_id[item["intention_name"]] = item["intention_id"]
                knowledge_infer_description[item["intention_name"]] = " ".join(item["llm_description"]) if item[
                    "llm_description"] else ""
                knowledge_type_lookup[item["intention_id"]] = item["knowledge_type"]
        """
        When use_llm is off or
        llm_threshold > 0 even if use_llm is on (meaning we still use the traditional approaches if user input is below this threshold),
        We initialize the matchers of these traditional approaches: keyword and semantic
        """
        knowledge_context = KnowledgeContext(
            knowledge=knowledge,
            infer_id=knowledge_infer_id,
            infer_description=knowledge_infer_description,
            type_lookup=knowledge_type_lookup,
        )

        # Load chatflow design data
        with open(chatflow_design_path, 'r', encoding='utf-8') as f:
            chatflow_design = json.load(f)

        # Load intentions
        with open(intention_path, 'r', encoding='utf-8') as f:
            intentions = json.load(f)

        # Load vector_db_collection
        with open(vector_db_collection_path, 'r', encoding='utf-8') as f:
            vector_db_collection = json.load(f)
        vector_db_collection_lookup = {v["node_id"]: v["db_collection_name"] for v in vector_db_collection}

        # Load dialog data
        with open(dialog_path, 'r', encoding='utf-8') as f:
            dialogs = json.load(f)
        dialog_lookup = {d["dialog_id"]: d for d in dialogs}

        return cls(
            agent_config=agent_config,
            knowledge_context=knowledge_context,
            chatflow_design=chatflow_design,
            intentions=intentions,
            dialog_lookup=dialog_lookup,
            vector_db_collection_lookup = vector_db_collection_lookup
        )