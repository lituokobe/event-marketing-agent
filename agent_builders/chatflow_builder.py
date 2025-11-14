from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from config.config_setup import ChatFlowConfig
from config.paths import VECTOR_DB_PATH
from elements.edge_initialization import create_edges
from elements.hang_up_node import hang_up
from elements.node_initialization import create_base_node, create_transfer_node
from functionals.log_utils import logger_chatflow
from functionals.matchers import KeywordMatcher, SemanticMatcher
from functionals.state import ChatState

def build_chatflow(chatflow_config: ChatFlowConfig):
    # Load the design and configuration for the chatflow
    agent_config = chatflow_config.agent_config
    knowledge_context = chatflow_config.knowledge_context
    # Set up matchers in knowledge_context
    """
    When use_llm is off or 
    llm_threshold > 0 even if use_llm is on (meaning we still use the traditional approaches if user input is below this threshold),
    We initialize the matchers of these traditional approaches: keyword and semantic
    """
    knowledge_keyword_matcher = None
    knowledge_semantic_matcher = None
    if not agent_config.use_llm or agent_config.llm_threshold > 0:
        # for intentions from knowledge
        knowledge_keyword_matcher = KeywordMatcher(knowledge_context.knowledge)
        knowledge_semantic_matcher = SemanticMatcher(
            "knowledge_collection",
            VECTOR_DB_PATH,
            knowledge_context.knowledge)

    knowledge_context.keyword_matcher=knowledge_keyword_matcher
    knowledge_context.semantic_matcher=knowledge_semantic_matcher

    chatflow_design = chatflow_config.chatflow_design
    intentions = chatflow_config.intentions
    dialog_lookup = chatflow_config.dialog_lookup
    vector_db_collection_lookup = chatflow_config.vector_db_collection_lookup

    # Create a look-up dict of main_flow_id -> starting_node_id.
    # This is helpful in identifying the structure of the chatflow design.
    starting_node_lookup = {
        flow.get("main_flow_id", None): flow.get("main_flow_content", {}).get("starting_node_id", None)
        for flow in chatflow_design.get("main_flows", [])
    }

    # Find the id of starting node of the starting main flow.
    # This node is special because it's triggered first without user's input.
    starting_main_flow_id = chatflow_design.get("starting_main_flow_id", None)
    if not isinstance(starting_main_flow_id, str):
        e_m = "初始主流程ID必须为字符串"
        logger_chatflow.error(e_m)
        raise TypeError(e_m)

    starting_node_id = starting_node_lookup.get(starting_main_flow_id, None)
    if not isinstance(starting_node_id, str):
        e_m = "初始节点ID必须为字符串"
        logger_chatflow.error(e_m)
        raise TypeError(e_m)

    # Router function to directly the conversation back to the last node in the state stack before assistant's response
    def route_to_workflow(state: ChatState) -> str:
        dialog_state = state.get("dialog_state", [])
        if not dialog_state:  # At the beginning, send to the first node
            return f"{starting_node_id}_reply"
        elif dialog_state[-1] == "hang_up":
            return END
        else:
            return dialog_state[-1]

    # TODO Start to build the Graph officially
    graph = StateGraph(ChatState)
    # Create hang_up node. It's better to be created first, other the factory functions later will
    # automatically build edges connected to this node.
    graph.add_node("hang_up", hang_up)

    # Iterate the main flows:
    for main_flow in chatflow_design.get("main_flows", []):
        if not main_flow:
            e_m = "设计不包含任何主流程"
            logger_chatflow.error(e_m)
            raise TypeError(e_m)

        main_flow_content = main_flow.get("main_flow_content", {})
        if not main_flow_content:
            e_m = "主流程不包含任何节点"
            logger_chatflow.error(e_m)
            raise TypeError(e_m)

        # Create base nodes:
        base_nodes = main_flow_content.get("base_nodes", [])

        for base_node in base_nodes:
            create_base_node(
                graph,
                main_flow["main_flow_id"],
                main_flow["main_flow_name"],
                base_node["node_id"],
                base_node["node_name"],
                base_node["default_reply_id_list"],
                vector_db_collection_lookup.get(base_node["node_id"], ""),
                base_node["intention_branches"],
                base_node["enable_logging"],
                agent_config,
                knowledge_context,
                intentions,
                dialog_lookup,
            )

        # Create transfer nodes
        transfer_nodes = main_flow_content.get("transfer_nodes", [])

        for transfer_node in transfer_nodes:
            create_transfer_node(
                graph,
                main_flow["main_flow_id"],
                main_flow["main_flow_name"],
                transfer_node["node_id"],
                transfer_node["node_name"],
                transfer_node["default_reply_id_list"],
                transfer_node["transfer_node_id"],
                transfer_node["enable_logging"],
                starting_node_lookup,
                dialog_lookup,
                agent_config
            )

        # Create the conditional edges from the base nodes
        edge_setups = main_flow_content.get("edge_setups", [])

        for edge_setup in edge_setups:
            create_edges(
                graph,
                main_flow["main_flow_id"],
                main_flow["main_flow_name"],
                edge_setup["node_id"],
                edge_setup["node_name"],
                edge_setup["node_edge_route"],
                edge_setup["enable_logging"],
                starting_node_lookup
            )

    # Create the conditional edges from the START node
    graph.add_conditional_edges(START, route_to_workflow)

    return graph.compile(checkpointer=MemorySaver())