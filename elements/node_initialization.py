from typing import Literal

from langgraph.constants import END
from langgraph.graph import StateGraph
from pymilvus import MilvusClient

from config.config_setup import NodeConfig, AgentConfig, KnowledgeContext, ChatflowDesignContext, GlobalConfigContext
from elements.intention_node import IntentionNode
from elements.reply_node import ReplyNode
from functionals.utils import update_target, next_main_flow
from functionals.log_utils import logger_chatflow

# factory function to create base node
def create_base_node(
    graph: StateGraph,
    main_flow: dict,
    main_flow_type: str,
    base_node: dict,
    agent_config: AgentConfig,
    knowledge_context: KnowledgeContext,
    global_config_context: GlobalConfigContext,
    intentions: list,
    milvus_client: MilvusClient | None = None,
):
    main_flow_id: str = main_flow.get("main_flow_id", "")
    main_flow_name: str = main_flow.get("main_flow_name", "")
    node_id: str = base_node.get("node_id", "")
    node_name: str = base_node.get("node_name", "")
    reply_content_info: list = base_node.get("reply_content_info", [])
    intention_branches: list = base_node.get("intention_branches", [])
    other_config: dict = base_node.get("other_config", {})
    enable_logging: bool = base_node.get("enable_logging", False)
    """
    Simulates user creating a base node via GUI.
    Adds the node set of one reply node and one intention node to the graph.
    This node can send pre-configured reply to the user first and then tell the intention from the user's next input.
    """
    # Create config (uses global defaults for shared paths via your paths.py or AgentConfig)
    config = NodeConfig(
        main_flow_id=main_flow_id,
        main_flow_name=main_flow_name,
        main_flow_type=main_flow_type,
        node_id=node_id,
        node_name=node_name,
        reply_content_info=reply_content_info,
        intention_branches=intention_branches,
        other_config=other_config,
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
    )
    intention_node = IntentionNode(
        config,
        knowledge_context=knowledge_context,
        global_config_context=global_config_context,
        intentions=intentions,
        milvus_client = milvus_client
    )

    # Add to graph
    graph.add_node(reply_node_name, reply_node)
    graph.add_node(intention_node_name, intention_node)
    if reply_content_info: #we stop for user to talk
        graph.add_edge(reply_node_name, END)
    else: #if no reply content se tup, we directly carry on to the intention node
        graph.add_edge(reply_node_name, intention_node_name)

    if enable_logging:
        log_info = (f"普通节点创立 - 主流程ID：{main_flow_id} - 主流程名称：{main_flow_name} - "
                    f"节点ID：{node_id} - 节点名称：{node_name}")
        logger_chatflow.info("系统消息：%s", log_info)

# factory function to create transfer node
def create_transfer_node(
    graph: StateGraph,
    main_flow: dict,
    main_flow_type: str,
    transfer_node: dict,
    agent_config: AgentConfig,
    chatflow_design_context: ChatflowDesignContext
):
    main_flow_id: str = main_flow.get("main_flow_id", "")
    main_flow_name: str = main_flow.get("main_flow_name", "")
    node_id: str = transfer_node.get("node_id", "")
    node_name: str = transfer_node.get("node_name", "")
    reply_content_info: list = transfer_node.get("reply_content_info", [])
    action: int = int(transfer_node.get("action", 0)) # 1-挂断 2-跳转下一主线流程 3-跳转指定主线流程
    transfer_node_id: str|None = transfer_node.get("master_process_id", "")
    other_config: dict = transfer_node.get("other_config", {})
    enable_logging: bool = transfer_node.get("enable_logging", False)
    starting_node_lookup: dict = chatflow_design_context.starting_node_lookup
    sort_lookup: dict = chatflow_design_context.sort_lookup
    """
    Simulates user creating a transfer node via GUI.
    Adds the node set of one reply node and one intention node to the graph.
    It only includes one reply sub-nodes, and gives 2 output: pre-configured reply and designated next-node in state.
    """
    # Create sub-node names
    reply_node_name = f"{node_id}_reply"

    # Identify the next node
    if action not in {1,2,3}:
        e_m = f"转换节点{node_id}-{node_name}执行动作无效"
        logger_chatflow.error(e_m)
        raise ValueError(e_m)
    if action == 1: # 挂断
        transfer_node_id = "hang_up"
    elif action == 2: # 跳转下一主线流程
        transfer_node_id = update_target(next_main_flow(main_flow_id, sort_lookup), starting_node_lookup)
    elif action == 3: # 跳转指定主线流程
        transfer_node_id = update_target(transfer_node_id, starting_node_lookup)

    # Create config (uses global defaults for shared paths via your paths.py or AgentConfig)
    config = NodeConfig(
        main_flow_id=main_flow_id,
        main_flow_name=main_flow_name,
        main_flow_type = main_flow_type,
        node_id=node_id,
        node_name=node_name,
        reply_content_info=reply_content_info,
        transfer_node_id=transfer_node_id,
        other_config=other_config,
        enable_logging=enable_logging,
        agent_config=agent_config
    )

    # Create sub-node instances
    reply_node = ReplyNode(config, next_node_name=transfer_node_id)
    
    # Add to graph
    graph.add_node(reply_node_name, reply_node)
    graph.add_edge(reply_node_name, transfer_node_id)

    if enable_logging:
        log_info = (f"转换节点创立 - 主流程ID：{main_flow_id} - 主流程名称：{main_flow_name} "
                    f"- 节点ID：{node_id} - 节点名称：{node_name}")
        logger_chatflow.info("系统消息：%s", log_info)

# factory function to create knowledge transfer node
def create_knowledge_reply_node(
        graph: StateGraph,
        knowledge_info: dict,
        agent_config: AgentConfig,
):
    node_id: str = knowledge_info.get("intention_id", "")
    node_name: str = knowledge_info.get("intention_name", "")
    other_config: dict = knowledge_info.get("other_config", {})

    reply_n = 0
    answer_list: list = knowledge_info.get("answer", [])
    if answer_list:
        selected_default_reply = answer_list[reply_n]
        reply_content_info = selected_default_reply.get("reply_content_info", [])
        action = selected_default_reply.get("action")

        # Validate action
        if action not in {1, 2, 3}: # 1-等待用户回复 2-挂断 3-跳转主流程
            e_m = f"知识库{node_id}-{node_name}执行动作无效"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)
    else:
        e_m = f"知识库{node_id}-{node_name}没有设定回答话术"
        logger_chatflow.error(e_m)
        raise ValueError(e_m)

    enable_logging: bool = knowledge_info.get("enable_logging", False)
    main_flow_id = "knowledge"
    main_flow_name = "知识库"
    reply_node_name = f"{node_id}_reply"

    config = NodeConfig(
        main_flow_id=main_flow_id,
        main_flow_name=main_flow_name,
        main_flow_type="knowledge_reply",
        node_id=node_id,
        node_name=node_name,
        reply_content_info=reply_content_info,
        other_config=other_config,
        enable_logging=enable_logging,
        agent_config=agent_config
    )

    if action == 1: # 等待用户回复
        reply_node = ReplyNode(config, next_node_name="pop") # We will get back to the previous node by default
        graph.add_node(reply_node_name, reply_node)
        graph.add_edge(reply_node_name, END) # End the flow to wait for user input
    elif action == 2: # 挂断
        reply_node = ReplyNode(config, next_node_name="hang_up")
        graph.add_node(reply_node_name, reply_node)
        graph.add_edge(reply_node_name, "hang_up")
    elif action == 3: # 指定主线流程
        reply_node = ReplyNode(config, next_node_name=None)
        graph.add_node(reply_node_name, reply_node)
        # This dynamic routing will be configured edge_initialization

    if enable_logging:
        log_info = (f"知识库节点创立 - 主流程ID：{main_flow_id} - 主流程名称：{main_flow_name} "
                    f"- 节点ID：{node_id} - 节点名称：{node_name}")
        logger_chatflow.info("系统消息：%s", log_info)

# factory function to create global config transfer node
def create_global_reply_node(
        graph: StateGraph,
        global_config: dict,
        agent_config: AgentConfig,
):
    context_type = int(global_config.get("context_type"))
    if context_type == 1:
        node_id = "no_input"
        node_name = "客户无应答"
    elif context_type == 2:
        node_id = "no_infer_result"
        node_name = "AI未识别"
    else:
        e_m = "全局配置不正确"
        logger_chatflow.error(e_m)
        raise ValueError(e_m)

    reply_n = 0
    answer_list: list = global_config.get("answer", [])
    if answer_list:
        selected_reply = answer_list[reply_n]
        reply_content_info = selected_reply.get("reply_content_info", [])
        action = selected_reply.get("action")

        # Validate action
        if action not in {1, 2, 3}: # 1-等待用户回复 2-挂断 3-跳转主流程
            e_m = f"全局配置{node_id}-{node_name}执行动作无效"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)
    else:
        e_m = f"全局配置{node_id}-{node_name}没有设定回答话术"
        logger_chatflow.error(e_m)
        raise ValueError(e_m)

    enable_logging: bool = global_config.get("enable_logging", False)
    main_flow_id = "global_config"
    main_flow_name = "全局配置"
    reply_node_name = f"{node_id}_reply"

    config = NodeConfig(
        main_flow_id=main_flow_id,
        main_flow_name=main_flow_name,
        main_flow_type="global_config_reply",
        node_id=node_id,
        node_name=node_name,
        reply_content_info=reply_content_info,
        # global config reply has no other_config
        enable_logging=enable_logging,
        agent_config=agent_config
    )

    if action == 1: # 等待用户回复
        reply_node = ReplyNode(config, next_node_name="pop") # We will get back to the previous node by default
        graph.add_node(reply_node_name, reply_node)
        graph.add_edge(reply_node_name, END) # End the flow to wait for user input
    elif action == 2: # 挂断
        reply_node = ReplyNode(config, next_node_name="hang_up")
        graph.add_node(reply_node_name, reply_node)
        graph.add_edge(reply_node_name, "hang_up")
    elif action == 3: # 指定主线流程
        reply_node = ReplyNode(config, next_node_name=None)
        graph.add_node(reply_node_name, reply_node)
        # This dynamic routing will be configured edge_initialization

    if enable_logging:
        log_info = (f"全局配置节点创立 - 主流程ID：{main_flow_id} - 主流程名称：{main_flow_name} "
                    f"- 节点ID：{node_id} - 节点名称：{node_name}")
        logger_chatflow.info("系统消息：%s", log_info)