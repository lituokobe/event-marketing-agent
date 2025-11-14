import time
from config.config_setup import NodeConfig
from functionals.log_utils import logger_chatflow
from functionals.matchers import KeywordMatcher, SemanticMatcher, LLMInferenceMatcher
from functionals.state import ChatState
from functionals.utils import get_last_user_message, intention_filter


# The class of the intention node
class IntentionNode:
    def __init__(self, config: NodeConfig, intentions: list):
        self.config = config

        # Get active intention ids to filter the intention
        # Get active intention id - intention branch id lookup table
        active_intention_ids: list[str|None] = []
        self.branch_id_lookup: dict[str, str] = {}
        self.branch_name_lookup: dict[str, str] = {}

        for branch in config.intention_branches:
            intention_ids = branch.get("intention_ids")
            if isinstance(intention_ids, list):
                for intention_id in intention_ids:
                    active_intention_ids.append(intention_id)
                    self.branch_id_lookup[intention_id] = branch["branch_id"]
                    self.branch_name_lookup[intention_id] = branch["branch_name"]

        if not active_intention_ids: # If there is no active intentions
            e_m = "没有任何意图需要创建"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)
        if len(active_intention_ids) != len(set(active_intention_ids)): # If there are duplicated intention ids
            e_m = "意图不能重复"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)

        filtered_intentions = intention_filter(intentions, active_intention_ids)

        if not filtered_intentions:
            e_m = "意图节点没有定义意图"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)

        # Initialize matchers
        if not config.agent_config.use_llm or self.config.agent_config.llm_threshold > 0:
            self.keyword_matcher = KeywordMatcher(config, filtered_intentions)
            self.semantic_matcher = SemanticMatcher(config, filtered_intentions)
        if config.agent_config.use_llm:
            self.llm_inference_matcher = LLMInferenceMatcher(config, filtered_intentions)

    def __call__(self, state: ChatState) -> dict:
        messages = state["messages"]
        previous_metadata = state["metadata"][-1]
        user_input = get_last_user_message(messages).lower()

        # Set up some output parameters
        type_id = ""
        # inference_type = ""
        # input_summary = ""
        branch_id = ""
        branch_name = "其他"

        # Identify user intention
        prev_time = time.time()

        # === User didn't say anything ===
        if len(user_input) == 0:
            next_state = "others"
            log_info = (f"用户没有说话 "
                        f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                        f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                        f"- 耗时：{(time.time() - prev_time):.3f}秒")
            user_logic_title = {"用户没有说话"},
            assistant_logic_title = f"【主线流程】{self.config.main_flow_name} -> ",
            # intention_approach = "用户没有说话"
            # i_a_details = "无"

        # === LLM Matching ===
        elif self.config.agent_config.use_llm and len(user_input) >= self.config.agent_config.llm_threshold:
            # LangChain LLM only takes argument of list of chat history: [{"role": "user", "content": "..."},
            # {"role": "assistant", "content": "..."}, ...]
            chat_history = messages
            # Keep the last N rounds of chat history specified by the client
            messages_to_keep = self.config.agent_config.llm_context_rounds*2
            if messages_to_keep <1: # if messages_to_keep is 0, only keep the last one round
                messages_to_keep = 1
            if len(messages) > messages_to_keep:
                chat_history = chat_history[-messages_to_keep:]
            type_id, inference_type, input_summary, _2 = self.llm_inference_matcher.llm_infer(chat_history)
            branch_id = self.branch_id_lookup.get(type_id, "others")
            branch_name = self.branch_name_lookup.get(type_id, "其他")

            if inference_type != "其他":
                next_state = branch_id
                log_info = (f"大模型意图命中 - 意图ID：{type_id} - 意图名称：{inference_type} "
                            f"- 意图分支ID：- {branch_id} - 意图分支名称：{branch_name}"
                            f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                            f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                            f"- 耗时：{(time.time() - prev_time):.3f}秒")
                user_logic_title = {f"主线流程【{branch_name}】分支 “{inference_type}”", f"大模型理解：“{input_summary}”"},
                assistant_logic_title = f"【主线流程】{self.config.main_flow_name} 、{branch_name} -> "
                # intention_approach = "大模型"
                # i_a_details = f"{type_id} - {inference_type} - {branch_id} - {branch_name}"
            else:
                next_state = "others"
                log_info = (f"没有意图命中(大模型开启) "
                            f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                            f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                            f"- 耗时：{(time.time() - prev_time):.3f}秒")
                user_logic_title = {f"没有意图命中(大模型开启)", f"大模型理解：“{input_summary}”"},
                assistant_logic_title = f"【主线流程】{self.config.main_flow_name} 、{branch_name} -> "
                # intention_approach = "没有意图命中(大模型开启)"
                # i_a_details = "无"

        else:
            # === Keyword Matching ===
            keyword_result = self.keyword_matcher.analyze_sentence(user_input)
            if len(keyword_result) > 0: # keywords identified
                type_id, inference_type, keywords, count = self.keyword_matcher.get_primary_type(keyword_result)
                branch_id = self.branch_id_lookup.get(type_id, "others")
                branch_name = self.branch_name_lookup.get(type_id, "其他")
                next_state = branch_id
                log_info = (
                            f"关键词意图命中 - {type_id} - {inference_type} - {count}个关键词：{'、'.join(keywords)} "
                            f"- 意图分支ID：- {branch_id} - 意图分支名称：{branch_name}"
                            f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                            f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                            f"- 耗时：{(time.time() - prev_time):.3f}秒")
                user_logic_title = {f"主线流程【{branch_name}】分支 “{inference_type}”", f"命中{count}个关键词：{'、'.join(keywords)}"},
                assistant_logic_title = f"【主线流程】{self.config.main_flow_name} 、{branch_name} -> "
                # intention_approach = "关键词"
                # i_a_details = f"{type_id} - {inference_type} - {count}个关键词：{'、'.join(keywords)} - {branch_id} - {branch_name}"

            # === Semantic Matching ===
            else:
                type_id, inference_type, content, cos_score = self.semantic_matcher.find_most_similar(user_input)
                if cos_score > self.config.agent_config.cosine_threshold:
                    branch_id = self.branch_id_lookup.get(type_id, "others")
                    branch_name = self.branch_name_lookup.get(type_id, "其他")
                    next_state = branch_id
                    log_info = (f"问法意图命中 - {type_id} - {inference_type} - 命中问法：{content} - 相似度：{cos_score:.3f} "
                                f"- 意图分支ID：- {branch_id} - 意图分支名称：{branch_name}"
                                f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                                f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                                f"- 耗时：{(time.time() - prev_time):.3f}秒")
                    user_logic_title = {f"主线流程【{branch_name}】分支 “{inference_type}”", f"命中问法：{content} - 相似度：{cos_score:.3f}"},
                    assistant_logic_title = f"【主线流程】{self.config.main_flow_name} 、{branch_name} -> "
                    # intention_approach = "问法"
                    # i_a_details = f"{type_id} - {inference_type} - 最相似问法：{content} - 相似度：{cos_score:.3f} - {branch_id} - {branch_name}"
                else:
                    next_state = "others"
                    log_info = (f"没有意图命中(大模型关闭) "
                                f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                                f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                                f"- 耗时：{(time.time() - prev_time):.3f}秒")
                    user_logic_title = {f"没有意图命中(大模型关闭)"},
                    assistant_logic_title = f"【主线流程】{self.config.main_flow_name} 、{branch_name} -> "
                    # intention_approach = "没有意图命中(大模型关闭)"
                    # i_a_details = "无"

        # Set up metadata
        updated_metadata = state["metadata"] + [
            {
                **previous_metadata,
                # "main_flow_id": self.config.main_flow_id,
                # "main_flow_name": self.config.main_flow_name,
                # "node_id": self.config.node_id,
                # "node_name": self.config.node_name,
                # "intention_approach": intention_approach,
                # "i_a_details": i_a_details,
                "role": "user",
                "content": user_input,
                "intention_tag": type_id,
                "logic":{
                    "user_logic_title": user_logic_title,
                    "assistant_logic_title": assistant_logic_title,
                    "detail": {
                        "main_flow_id": self.config.main_flow_id,
                        "node_id": self.config.node_id,
                        "branch_id": branch_id,
                        "infer_type": "1 意图 2 知识库",
                        "infer_use_type": "知识库的类型1通用问题 2业务问题 3一般问题"
                    }
                }
            }
        ]

        # Log information
        if self.config.enable_logging:
            logger_chatflow.info("系统消息：%s", log_info)

        return {
            "messages": messages,
            "dialog_state": next_state,
            "metadata": updated_metadata
        }