from langgraph.constants import END
from langgraph.graph import StateGraph
from config.config_setup import NodeConfig, AgentConfig, KnowledgeContext
from elements.intention_node import IntentionNode
from elements.knowledge_node import KnowledgeNode
from elements.reply_node import ReplyNode
from functionals.log_utils import logger_chatflow
from functionals.matchers import KeywordMatcher, SemanticMatcher
from functionals.utils import update_target


# factory function to create base node
def create_base_node(
    graph: StateGraph,
    main_flow_id: str,
    main_flow_name: str,
    node_id: str,
    node_name: str,
    default_reply_id_list: list,
    db_collection_name: str,
    intention_branches: list[dict|None],
    enable_logging: bool,
    agent_config: AgentConfig,
    knowledge_context: KnowledgeContext,
    intentions: list[dict|None],
    dialog_lookup: dict[str, dict]
):
    """
    Simulates user creating a base node via GUI.
    Adds the node set of one reply node and one intention node to the graph.
    This node can send pre-configured reply to the user first and then tell the intention from the user's next input.
    """
    # Create config (uses global defaults for shared paths via your paths.py or AgentConfig)
    config = NodeConfig(
        main_flow_id=main_flow_id,
        main_flow_name=main_flow_name,
        node_id=node_id,
        node_name=node_name,
        default_reply_id_list=default_reply_id_list,
        db_collection_name=db_collection_name,
        intention_branches=intention_branches,
        enable_logging=enable_logging,
        agent_config=agent_config,
    )

    # Create sub-node names
    reply_node_name = f"{node_id}_reply"
    intention_node_name = f"{node_id}_intention"

    # Create sub-node instances
    reply_node = ReplyNode(
        config,
        next_node_name=intention_node_name,
        dialog_lookup=dialog_lookup
    )
    intention_node = IntentionNode(
        config,
        knowledge_context=knowledge_context,
        intentions=intentions,
    )

    # Add to graph
    graph.add_node(reply_node_name, reply_node)
    graph.add_node(intention_node_name, intention_node)
    if default_reply_id_list: #if default reply is input, we stop for user to talk
        graph.add_edge(reply_node_name, END)
    else: #if no default reply, we directly carry on to the intention node
        graph.add_edge(reply_node_name, intention_node_name)

    if enable_logging:
        log_info = (f"普通节点创立 - 主流程ID：{main_flow_id} - 主流程名称：{main_flow_name} - "
                    f"节点ID：{node_id} - 节点名称：{node_name}")
        logger_chatflow.info("系统消息：%s", log_info)

# factory function to create transfer node
def create_transfer_node(
    graph: StateGraph,
    main_flow_id: str,
    main_flow_name: str,
    node_id: str,
    node_name: str,
    default_reply_id_list: list,
    transfer_node_id: str,
    enable_logging: bool,
    starting_node_lookup: dict[str, str],
    dialog_lookup: dict[str, dict],
    agent_config: AgentConfig,
):
    """
    Simulates user creating a transfer node via GUI.
    Adds the node set of one reply node and one intention node to the graph.
    It only includes one reply sub-nodes, and gives 2 output: pre-configured reply and designated next-node in state.
    """
    # Create config (uses global defaults for shared paths via your paths.py or AgentConfig)
    config = NodeConfig(
        main_flow_id=main_flow_id,
        main_flow_name=main_flow_name,
        node_id=node_id,
        node_name=node_name,
        default_reply_id_list=default_reply_id_list,
        transfer_node_id=transfer_node_id,
        enable_logging=enable_logging,
        agent_config=agent_config
    )

    # Create sub-node names
    reply_node_name = f"{node_id}_reply"

    # if the node to transfer is a regular node, the first sub node must be the reply node
    # if the node to transfer is "hang_up", just simply directs to it
    # the transfer_node_id can be the main_flow_id if switching main flow, need to convert it
    if transfer_node_id != "hang_up":
        transfer_node_id = update_target(transfer_node_id, starting_node_lookup)

    # Create sub-node instances
    reply_node = ReplyNode(config, next_node_name=transfer_node_id, dialog_lookup=dialog_lookup)
    
    # Add to graph
    graph.add_node(reply_node_name, reply_node)

    if transfer_node_id != "hang_up" and default_reply_id_list: #If next node is not "hang_up" and there ia a default reply, we stop for user to talk
        graph.add_edge(reply_node_name, END)
    else:
        graph.add_edge(reply_node_name, transfer_node_id)

    if enable_logging:
        log_info = (f"转换节点创立 - 主流程ID：{main_flow_id} - 主流程名称：{main_flow_name} "
                    f"- 节点ID：{node_id} - 节点名称：{node_name}")
        logger_chatflow.info("系统消息：%s", log_info)