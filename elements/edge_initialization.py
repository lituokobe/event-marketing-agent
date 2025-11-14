from langgraph.graph import StateGraph

from functionals.log_utils import logger_chatflow
from functionals.state import ChatState
from functionals.utils import update_target


# factory function to create router and conditional edges
def create_edges(
        graph: StateGraph,
        main_flow_id: str,
        main_flow_name: str,
        node_id: str,
        node_name: str,
        route_map: dict[str, str],
        enable_logging: bool,
        starting_node_lookup: dict[str, str]
):
    intention_node_id = f"{node_id}_intention"

    # Update the target ids in the route map
    updated_route_map = {k: update_target(v, starting_node_lookup) for k, v in route_map.items()}

    def route_func(state: ChatState) -> str:
        dialog_state = state.get("dialog_state", [])
        if dialog_state:
            return updated_route_map[dialog_state[-1]]
        else:
            return updated_route_map["others"]

    graph.add_conditional_edges(intention_node_id, route_func)

    if enable_logging:
        log_info = (f"普通节点条件边创立 - 主流程ID：{main_flow_id} - 主流程名称：{main_flow_name} "
                    f"- 节点ID：{node_id} - 节点名称：{node_name}")
        logger_chatflow.info("系统消息：%s", log_info)