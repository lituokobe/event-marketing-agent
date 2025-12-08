from langgraph.graph import StateGraph
from config.config_setup import ChatflowDesignContext, KnowledgeContext
from functionals.log_utils import logger_chatflow
from functionals.state import ChatState
from functionals.utils import update_target

# factory function to create router and conditional edges
def create_edges(
        graph: StateGraph,
        main_flow: dict,
        edge_setup: dict,
        chatflow_design_context: ChatflowDesignContext,
        knowledge_context: KnowledgeContext,
):
    main_flow_id: str = main_flow.get("main_flow_id", "")
    main_flow_name: str = main_flow.get("main_flow_name", "")
    node_id: str = edge_setup.get("node_id", "")
    node_name: str = edge_setup.get("node_name", "")
    route_map: dict = edge_setup.get("route_map", {})
    enable_logging: bool = edge_setup.get("enable_logging", False)
    intention_node_id = f"{node_id}_intention"
    starting_node_lookup: str = chatflow_design_context.starting_node_lookup
    knowledge_multi_round_lookup: dict = knowledge_context.multi_round_lookup

    # Update the target ids in the route map
    updated_route_map = {k: update_target(v, starting_node_lookup) for k, v in route_map.items()}
    local_node_intentions = list(updated_route_map.keys())

    def route_func(state: ChatState) -> str:
        dialog_state = state.get("dialog_state", [])
        if not dialog_state:
            return updated_route_map["others"]
        last_dialog_state = dialog_state[-1] if dialog_state else ""

        if not last_dialog_state:
            return updated_route_map["others"]
        elif last_dialog_state in local_node_intentions: # if the intention is specified by users in the local node configuration
            return updated_route_map[last_dialog_state]
        elif last_dialog_state in knowledge_multi_round_lookup:
            # if the intention is global knowledge with multi-round reply
            # knowledge intention id->knowledge main flow if->first node if of the main flow
            return update_target(knowledge_multi_round_lookup[last_dialog_state], starting_node_lookup)
        else: # if the intention is global knowledge with single round reply
            return update_target(last_dialog_state, starting_node_lookup)

    graph.add_conditional_edges(intention_node_id, route_func)

    if enable_logging:
        log_info = (f"普通节点条件边创立 - 主流程ID：{main_flow_id} - 主流程名称：{main_flow_name} "
                    f"- 节点ID：{node_id} - 节点名称：{node_name}")
        logger_chatflow.info("系统消息：%s", log_info)

# factory function to create router and conditional edges for knowledge
def create_knowledge_edges(
        graph: StateGraph,
        knowledge_info: dict,
        chatflow_design_context: ChatflowDesignContext,
):
    node_id: str = knowledge_info.get("intention_id", "")
    node_name: str = knowledge_info.get("intention_name", "")
    enable_logging: bool = knowledge_info.get("enable_logging", False)
    main_flow_id = "knowledge"
    main_flow_name = "知识库"
    reply_node_name = f"{node_id}_reply"
    reply_n = 0
    answer_list: list = knowledge_info.get("answer", [])
    if answer_list:
        selected_default_reply = answer_list[reply_n]
        action = selected_default_reply.get("action")
    else:
        e_m = f"知识库{node_id}-{node_name}没有设定回答话术"
        logger_chatflow.error(e_m)
        raise ValueError(e_m)

    starting_node_lookup: dict = chatflow_design_context.starting_node_lookup
    mf_node_ids: set = chatflow_design_context.mf_node_ids
    mf_starting_node_ids: set = chatflow_design_context.mf_starting_node_ids
    starting_node_id: str = chatflow_design_context.starting_node_id

    # We only set up edges that lead to main flows here
    if action == 3: # 指定主线流程
        transfer_node_id = selected_default_reply.get("next")
        if transfer_node_id not in {-1, -2, 3}: # -1-原主线节点 -2-原主线流程 3-指定主线流程
            e_m = f"知识库{node_id}-{node_name}指定主线流程无效"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)

        def route_func(state: ChatState) -> str:
            dialog_state = state.get("dialog_state", [])
            if not dialog_state:
                return f"{starting_node_id}_reply" #If no history states, just go to the very first node

            # Find the last intention node and main flow id.
            if transfer_node_id == -1 : # 原主线节点
                last_mf_node = None
                for item in reversed(dialog_state):
                    # Update last_intention_node if not already found
                    item_no_suffix = item.removesuffix("_intention")
                    if last_mf_node is None and item_no_suffix in mf_node_ids:
                        last_mf_node = item_no_suffix
                        # Early exit
                        break
                last_mf_node = last_mf_node or starting_node_id
                return f"{last_mf_node}_reply" # go to the base node's reply sub node
            elif transfer_node_id == -2: #原主线流程
                last_mf_starting_node = None
                for item in reversed(dialog_state):
                    item_no_suffix = item.removesuffix("_intention")
                    # Update last_main_flow_id if not already found
                    if last_mf_starting_node is None and item_no_suffix in mf_starting_node_ids:
                        last_mf_starting_node = item_no_suffix
                        break
                last_mf_starting_node = last_mf_starting_node or starting_node_id
                return f"{last_mf_starting_node}_reply" # go to the first node of the main flow
            else: # 3- 指定主线流程
                master_process_id = selected_default_reply.get("master_process_id")
                return update_target(master_process_id, starting_node_lookup) # go to the specified node/main flow

        graph.add_conditional_edges(reply_node_name, route_func)

        if enable_logging:
            log_info = (f"知识库节点条件边创立 - 主流程ID：{main_flow_id} - 主流程名称：{main_flow_name} "
                        f"- 节点ID：{node_id} - 节点名称：{node_name} - ")
            logger_chatflow.info("系统消息：%s", log_info)

# factory function to create router and conditional edges for global configs
def create_global_edges(
        graph: StateGraph,
        global_config: dict,
        chatflow_design_context: ChatflowDesignContext,
):
    context_type = int(global_config.get("context_type", 0))
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

    enable_logging: bool = global_config.get("enable_logging", False)
    main_flow_id = "global_config"
    main_flow_name = "全局配置"
    reply_node_name = f"{node_id}_reply"

    reply_n = 0
    answer_list: list = global_config.get("answer", [])
    if answer_list:
        selected_default_reply = answer_list[reply_n]
        action = selected_default_reply.get("action")
    else:
        e_m = f"全局配置{node_id}-{node_name}没有设定回答话术"
        logger_chatflow.error(e_m)
        raise ValueError(e_m)

    starting_node_lookup: dict = chatflow_design_context.starting_node_lookup
    mf_node_ids: set = chatflow_design_context.mf_node_ids
    mf_starting_node_ids: set = chatflow_design_context.mf_starting_node_ids
    starting_node_id: str = chatflow_design_context.starting_node_id

    # We only set up edges that lead to main flows here
    if action == 3: # 指定主线流程
        transfer_node_id = selected_default_reply.get("next")
        if transfer_node_id not in {-1, -2, 3}: # -1-原主线节点 -2-原主线流程 3-指定主线流程
            e_m = f"全局配置{node_id}-{node_name}指定主线流程无效"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)

        def route_func(state: ChatState) -> str:
            dialog_state = state.get("dialog_state", [])
            if not dialog_state:
                return f"{starting_node_id}_reply" #If no history states, just go to the very first node

            # Find the last intention node and main flow id.
            if transfer_node_id == -1 : # 原主线节点
                last_mf_node = None
                for item in reversed(dialog_state):
                    # Update last_intention_node if not already found
                    item_no_suffix = item.removesuffix("_intention")
                    if last_mf_node is None and item_no_suffix in mf_node_ids:
                        last_mf_node = item_no_suffix
                        # Early exit
                        break
                last_mf_node = last_mf_node or starting_node_id
                return f"{last_mf_node}_reply" # go to the base node's reply sub node
            elif transfer_node_id == -2: #原主线流程
                last_mf_starting_node = None
                for item in reversed(dialog_state):
                    item_no_suffix = item.removesuffix("_intention")
                    # Update last_main_flow_id if not already found
                    if last_mf_starting_node is None and item_no_suffix in mf_starting_node_ids:
                        last_mf_starting_node = item_no_suffix
                        break
                last_mf_starting_node = last_mf_starting_node or starting_node_id
                return f"{last_mf_starting_node}_reply" # go to the first node of the main flow
            else: # 3- 指定主线流程
                master_process_id = selected_default_reply.get("master_process_id")
                return update_target(master_process_id, starting_node_lookup) # go to the specified node/main flow

        graph.add_conditional_edges(reply_node_name, route_func)

        if enable_logging:
            log_info = (f"全局配置节点条件边创立 - 主流程ID：{main_flow_id} - 主流程名称：{main_flow_name} "
                        f"- 节点ID：{node_id} - 节点名称：{node_name}")
            logger_chatflow.info("系统消息：%s", log_info)