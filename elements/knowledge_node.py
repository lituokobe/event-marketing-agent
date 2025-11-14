import json
import time
from config.config_setup import NodeConfig
from functionals.log_utils import logger_chatflow
from functionals.matchers import KeywordMatcher, SemanticMatcher, LLMInferenceMatcher
from functionals.state import ChatState
from functionals.utils import get_last_user_message, intention_filter


# The class of the knowledge node
class KnowledgeNode:
    def __init__(self, config: NodeConfig, knowledge: list, next_node_name:str, dialog_lookup: dict):
        self.config = config
        self.knowledge = knowledge
        self.next_node_name = next_node_name
        self.dialog_lookup = dialog_lookup

        if not isinstance(knowledge, list):
            e_m = "知识数据必须是一个列表"
            logger_chatflow.error(e_m)
            raise TypeError(e_m)

        # Set up "end_call" in metadata
        self.end_call = False
        if self.next_node_name == "hang_up":
            self.end_call = True

        # Initialize matchers
        self.keyword_matcher = KeywordMatcher(config, knowledge)
        self.semantic_matcher = SemanticMatcher(config, knowledge)
        if config.use_llm:
            self.llm_inference_matcher = LLMInferenceMatcher(config, knowledge)

    def __call__(self, state: ChatState) -> dict:
        user_input = get_last_user_message(state["messages"]).lower()
        messages = state["messages"]
        previous_metadata = state["metadata"][-1]
        dialog_id = previous_metadata.get("dialog_id", "")
        content = previous_metadata.get("content", "")

        # Identify user intention
        # === User didn't say anything ===
        if len(user_input) == 0:
            log_info = (f"用户没有说话 - 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} - "
                        f"节点ID：{self.config.node_id} - 节点名称：{self.config.node_name}")
            intention_approach = "用户没有说话"
            i_a_details = "无"

        else:
            prev_time = time.time()

            # === Keyword Matching ===
            keyword_result = self.keyword_matcher.analyze_sentence(user_input)
            if len(keyword_result) > 0: # keywords identified
                primary_id, keyword_type, keywords, count = self.keyword_matcher.get_primary_type(keyword_result)
                dialog_id = primary_id
                content = self.dialog_lookup.get(dialog_id, {}).get('content', "")
                messages = messages + [{"role": "assistant", "content": content}]

                log_info = (
                            f"知识库关键词意图命中 - {primary_id} - {keyword_type} - {count}个关键词：{'、'.join(keywords)} "
                            f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                            f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                            f"- 耗时：{(time.time() - prev_time):.3f}秒")
                intention_approach = "知识库关键词"
                i_a_details = f"{primary_id} - {keyword_type} - {count}个关键词：{'、'.join(keywords)}"

            # === Semantic Matching ===
            else:
                type_id, semantic_type, content, cos_score = self.semantic_matcher.find_most_similar(user_input)
                if cos_score > self.config.cosine_threshold:
                    dialog_id = type_id
                    content = self.dialog_lookup.get(dialog_id, {}).get('content', "")
                    messages = messages + [{"role": "assistant", "content": content}]

                    log_info = (f"知识库问法意图命中 - {type_id} - {semantic_type} - 最相似问法：{content} - 余弦相似度：{cos_score:.3f} "
                                f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                                f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                                f"- 耗时：{(time.time() - prev_time):.3f}秒")
                    intention_approach = "知识库问法"
                    i_a_details = f"{type_id} - {semantic_type} - 最相似问法：{content} - 余弦相似度：{cos_score:.3f}"

                # === LLM Fallback ===
                elif self.config.use_llm:
                    type_id, inference_type, _1, _2 = self.llm_inference_matcher.llm_infer(user_input)

                    if inference_type != "其他":
                        dialog_id = type_id
                        content = self.dialog_lookup.get(dialog_id, {}).get('content', "")
                        messages = messages + [{"role": "assistant", "content": content}]

                        log_info = (f"知识库大模型意图命中 - {type_id} - {inference_type} "
                                    f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                                    f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                                    f"- 耗时：{(time.time() - prev_time):.3f}秒")
                        intention_approach = "知识库大模型"
                        i_a_details = f"{type_id} - {inference_type}"
                    else:
                        log_info = (f"没有意图命中(大模型开启) "
                                    f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                                    f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                                    f"- 耗时：{(time.time() - prev_time):.3f}秒")
                        intention_approach = "没有意图命中(大模型开启)"
                        i_a_details = "无"

                # === LLM is disabled ===
                else:
                    log_info = (f"没有意图命中(大模型关闭) "
                                f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                                f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                                f"- 耗时：{(time.time() - prev_time):.3f}秒")
                    intention_approach = "没有意图命中(大模型关闭)"
                    i_a_details = "无"

        # Set up metadata
        updated_metadata = state["metadata"] + [
            {
                **previous_metadata,
                "dialog_id": dialog_id,
                "content": content,
                "main_flow_id": self.config.main_flow_id,
                "main_flow_name": self.config.main_flow_name,
                "node_id": self.config.node_id,
                "node_name": self.config.node_name,
                "intention_approach": intention_approach,
                "i_a_details": i_a_details,
                "end_call": self.end_call
            }
        ]

        # Log information
        if self.config.enable_logging:
            logger_chatflow.info("系统消息：%s", log_info)

        return {
            "messages": messages,
            "dialog_state": self.next_node_name,
            "metadata": updated_metadata
        }