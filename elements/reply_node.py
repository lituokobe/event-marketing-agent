import copy
from langchain_core.messages import AIMessage
from config.config_setup import NodeConfig
from data.string_asset import infer_tool_str
from functionals.log_utils import logger_chatflow
from functionals.state import ChatState
from functionals.utils import process_reply, get_last_user_log, get_logs_from_last_user, get_last_user_message

# This node will respond the user based on the pre-configured reply
class ReplyNode:
    def __init__(self, config: NodeConfig, next_node_name:str|None):
        self.config = config
        self.next_node_name = next_node_name

        # Set up "end_call" in metadata
        self.end_call = False
        if self.next_node_name == "hang_up":
            self.end_call = True

    def __call__(self, state: ChatState) -> dict:
        # Get the message and last user message
        messages = state.get("messages", [])
        user_input = get_last_user_message(messages).lower()
        print(f"当前{self.config.node_id}_reply-{self.config.node_name}节点开始工作")

        # Get the reply content, select the first item from reply_content_info
        reply_n = 0
        dialog_id, reply_content = "", ""
        if self.config.reply_content_info:
            dialog_id, reply_content = process_reply(self.config.reply_content_info[reply_n], user_input)

        # Get the last log info
        logs = state.get("logs", [])
        previous_log = logs[-1] if logs else {}

        # Get the last metadata
        metadata = state.get("metadata", [])
        previous_metadata = metadata[-1] if metadata else {}

        # Get the last logic
        previous_logic = previous_metadata.get("logic", {})

        # Get the last complete process
        previous_complete_process = previous_logic.get("complete_process", [])

        # Update the complete_process
        updated_complete_process = copy.deepcopy(previous_complete_process)
        # Decide whether the main flow is finished by identifying if the node is a transfer node in a main flow
        if self.config.transfer_node_id and self.config.main_flow_type == "regular":  # It's a transfer node
            updated_complete_process.append(self.config.main_flow_id)

        # Setup log info and reply message
        if reply_content:
            updated_messages = messages + [AIMessage(content=reply_content)]
            updated_logs = logs + [{
                **previous_log,
                "role": "assistant",
                "content": reply_content,
                "main_flow_id": self.config.main_flow_id,
                "main_flow_name": self.config.main_flow_name,
                "node_id": self.config.node_id,
                "node_name": self.config.node_name,
                "other_config": self.config.other_config
            }]

            # Create user_logic_title
            last_user_log = get_last_user_log(logs) or {}
            if not last_user_log:
                user_logic_title = {}
            else:
                match_to = last_user_log.get("match_to", "")
                infer_tool = last_user_log.get("infer_tool", "")
                if match_to == "主线流程":
                    if infer_tool == "大模型":
                        user_logic_title = {
                            "匹配到": match_to,
                            "匹配到分支": f"【{last_user_log.get('branch_name','')}】-【{last_user_log.get('intention_name','')}】",
                            "匹配方式": f"【{infer_tool}】",
                            "大模型理解": last_user_log.get('llm_input_summary', '')
                        }
                    elif infer_tool == "关键词":
                        user_logic_title = {
                            "匹配到": match_to,
                            "匹配到分支": f"【{last_user_log.get('branch_name', '')}】-【{last_user_log.get('intention_name', '')}】",
                            "匹配方式": f"【{infer_tool}】",
                            "匹配内容": f"【{last_user_log.get('matching_content')}】",
                            "匹配关键词数量": int(last_user_log.get('matching_score', 0))
                        }
                    elif infer_tool == "问法":
                        user_logic_title = {
                            "匹配到": match_to,
                            "匹配到分支": f"【{last_user_log.get('branch_name', '')}】-【{last_user_log.get('intention_name', '')}】",
                            "匹配方式": f"【{infer_tool}】",
                            "匹配内容": f"【{last_user_log.get('matching_content')}】",
                            "匹配分数": last_user_log.get('matching_score', 0.0)
                        }
                    else:
                        e_m = f"主线流程推理工具有误"
                        logger_chatflow.error(e_m)
                        raise ValueError(e_m)
                elif match_to == "知识库":
                    if infer_tool == "大模型":
                        user_logic_title = {
                            "匹配到": match_to,
                            "匹配到分支": f"【{last_user_log.get('intention_name', '')}】",
                            "匹配方式": f"【{infer_tool}】",
                            "知识库类型": f"【{last_user_log.get('knowledge_type')}】",
                            "大模型理解": last_user_log.get('llm_input_summary', ''),
                        }
                    elif infer_tool == "关键词":
                        user_logic_title = {
                            "匹配到": match_to,
                            "匹配到分支": f"【{last_user_log.get('intention_name', '')}】",
                            "匹配方式": f"【{infer_tool}】",
                            "知识库类型": f"【{last_user_log.get('knowledge_type')}】",
                            "匹配内容": f"【{last_user_log.get('matching_content')}】",
                            "匹配关键词数量": int(last_user_log.get('matching_score', 0))
                        }
                    elif infer_tool == "问法":
                        user_logic_title = {
                            "匹配到": match_to,
                            "匹配到分支": f"【{last_user_log.get('intention_name', '')}】",
                            "匹配方式": f"【{infer_tool}】",
                            "知识库类型": f"【{last_user_log.get('knowledge_type')}】",
                            "匹配内容": f"【{last_user_log.get('matching_content')}】",
                            "匹配分数": last_user_log.get('matching_score', 0.0)
                        }
                    else:
                        e_m = f"知识库推理工具有误"
                        logger_chatflow.error(e_m)
                        raise ValueError(e_m)
                elif match_to == "没有意图命中":
                    if infer_tool in set(infer_tool_str):
                        user_logic_title = {
                            "匹配到": match_to,
                            "匹配方式": f"【{infer_tool}】",
                        }
                    else:
                        e_m = f"没有意图命中推理工具有误"
                        logger_chatflow.error(e_m)
                        raise ValueError(e_m)
                else:
                    e_m = f"用户log匹配流程有误"
                    logger_chatflow.error(e_m)
                    raise ValueError(e_m)

            # Create assistant_logic_title
            if self.config.main_flow_name == "知识库":
                assistant_logic_title = f"【知识库】:{self.config.main_flow_name}、{self.config.node_name}"
            else:
                assistant_logic_title = f"【主线流程】:{self.config.main_flow_name}、{self.config.node_name}"

            updated_metadata = metadata + [{
                **previous_metadata,
                "role": "assistant",
                "content": self.config.reply_content_info[reply_n],
                "dialog_id": dialog_id,
                "end_call": self.end_call,
                "logic":{
                    "user_logic_title":user_logic_title,
                    "assistant_logic_title":assistant_logic_title,
                    "complete_process": updated_complete_process,
                    "detail":get_logs_from_last_user(updated_logs)
                },
                "other_config": self.config.other_config
            }]
        else:
            # When there is no default reply set up
            updated_messages = messages + [AIMessage(content="")]
            updated_logs = logs + [{
                **previous_log,
                "role": "assistant",
                "content": "",
                "main_flow_id": self.config.main_flow_id,
                "main_flow_name": self.config.main_flow_name,
                "node_id": self.config.node_id,
                "node_name": self.config.node_name,
                "other_config": self.config.other_config
            }]
            # We will not add any extra item to meatadata, but need to update complete_process if this is a transfer node of a regular main flow
            updated_metadata = copy.deepcopy(metadata)
            if updated_metadata:
                if "logic" not in updated_metadata[-1]:
                    updated_metadata[-1]["logic"] = {}
                updated_metadata[-1]["logic"]["complete_process"] = updated_complete_process

        # Log information
        if self.config.enable_logging:
            if updated_logs[-1]:
                logger_chatflow.info("系统消息：%s", updated_logs[-1])

        print(f"{self.config.node_id}_reply-{self.config.node_name}节点完成工作")
        return {
            "messages": updated_messages,
            "dialog_state": self.next_node_name,
            "logs": updated_logs,
            "metadata":updated_metadata
        }