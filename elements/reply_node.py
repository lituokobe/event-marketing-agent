from config.config_setup import NodeConfig
from functionals.log_utils import logger_chatflow
from functionals.state import ChatState

# This node will respond the user based on the pre-configured reply
class ReplyNode:
    def __init__(self, config: NodeConfig, next_node_name:str, dialog_lookup: dict):
        self.config = config
        self.next_node_name = next_node_name

        # Get the default reply content, select the first id from default_reply_id_list
        reply_n = 0
        self.default_reply_id = self.config.default_reply_id_list[reply_n] if self.config.default_reply_id_list else ""
        self.default_reply = dialog_lookup.get(self.default_reply_id, {}).get("content", "")

        # Set up "end_call" in metadata
        self.end_call = False
        if self.next_node_name == "hang_up":
            self.end_call = True

    def __call__(self, state: ChatState) -> dict:
        messages = state["messages"]
        previous_metadata = state["metadata"][-1]

        # Setup log info and reply message
        if self.default_reply_id:
            log_info = (f"发送预设话术 - 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} - "
                        f"节点ID：{self.config.node_id} - 节点名称：{self.config.node_name}")
            messages = messages + [{"role": "assistant", "content": self.default_reply}]
        else:
            log_info = (f"没有预设话术 - 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} - "
                        f"节点ID：{self.config.node_id} - 节点名称：{self.config.node_name}")

        # update metadata
        updated_metadata = state["metadata"] + [
            {
                **previous_metadata,
                "role": "assistant",
                "content": self.default_reply,
                "dialog_id": self.default_reply_id,
                "end_call": self.end_call
            }
        ]
        assistant_logic_title = f"{previous_metadata.get('logic',{}).get('assistant_logic_title','')} {self.config.main_flow_name} {self.config.node_name}"
        updated_metadata[-1]["logic"]["assistant_logic_title"]= assistant_logic_title

        # Log information
        if self.config.enable_logging:
            logger_chatflow.info("系统消息：%s", log_info)

        return {
            "messages": messages,
            "dialog_state": self.next_node_name,
            "metadata":updated_metadata
        }