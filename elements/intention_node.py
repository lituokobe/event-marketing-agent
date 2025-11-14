import time
from config.config_setup import NodeConfig, KnowledgeContext
from functionals.integrated_matchers import integrated_keywords_matcher, integrated_semantic_matcher
from functionals.log_utils import logger_chatflow
from functionals.matchers import KeywordMatcher, SemanticMatcher, LLMInferenceMatcher
from functionals.state import ChatState
from functionals.utils import get_last_user_message, intention_filter

# The class of the intention node
class IntentionNode:
    def __init__(self,
                 config: NodeConfig,
                 knowledge_context: KnowledgeContext,
                 intentions: list,
                 ):
        self.config = config
        self.knowledge_type_lookup = knowledge_context.type_lookup

        # Get active intention ids to filter the intention
        # Get active intention id - intention branch id lookup table
        self.branch_id_lookup, self.branch_type_lookup, active_intention_ids = {}, {}, []

        # create active_intention_ids
        # create lookup table to look for branch_id and branch_type with intention_id
        for branch in config.intention_branches:
            intention_ids = branch.get("intention_ids", [])
            if isinstance(intention_ids, list):
                for intention_id in intention_ids:
                    if intention_id in self.branch_id_lookup:
                        e_m = f"意图不能重复"
                        logger_chatflow.error(e_m)
                        raise ValueError(e_m)
                    active_intention_ids.append(intention_id)
                    self.branch_id_lookup[intention_id] = branch["branch_id"]
                    self.branch_type_lookup[intention_id] = branch["branch_type"]

        if not active_intention_ids: # If there is no active intentions
            e_m = "没有任何意图需要创建"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)

        filtered_intentions = intention_filter(intentions, active_intention_ids)

        if not filtered_intentions:
            e_m = "意图节点没有定义意图"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)

        # Initialize matchers
        """
        When use_llm is off or 
        llm_threshold > 0 even if use_llm is on (meaning we still use the traditional approaches if user input is below this threshold),
        We initialize the matchers of these traditional approaches: keyword and semantic
        """
        if not config.agent_config.use_llm or config.agent_config.llm_threshold > 0:
            # for intentions specified in this node
            self.keyword_matcher = KeywordMatcher(filtered_intentions)
            self.semantic_matcher = SemanticMatcher(
                config.db_collection_name,
                config.agent_config.vector_db_path,
                filtered_intentions)
            # for intentions from knowledge
            self.knowledge_keyword_matcher = knowledge_context.keyword_matcher
            self.knowledge_semantic_matcher = knowledge_context.semantic_matcher

        if config.agent_config.use_llm:
            self.llm_matcher = LLMInferenceMatcher(config,
                                                   filtered_intentions,
                                                   knowledge_context.infer_id,
                                                   knowledge_context.infer_description,
                                                   config.agent_config.intention_priority)

    def __call__(self, state: ChatState) -> dict:
        messages = state["messages"]
        previous_metadata = state["metadata"][-1]
        user_input = get_last_user_message(messages).lower()
        branch_count = previous_metadata.get("branch_count", {})
        prev_time = time.time()

        # Set up default outputs
        type_id, branch_id, branch_type = "", "", "其他"
        inference_type, inference_use_type, next_state = "无", "无", "others"

        # Identify user intention
        # === Case 1: Empty input ===
        if not user_input:
            next_state = "others"
            log_info = (
                f"用户没有说话 "
                f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                f"- 耗时：{(time.time() - prev_time):.3f}秒"
            )
            user_logic_title = {"用户没有说话"}
            assistant_logic_title = f"【主线流程】{self.config.main_flow_name} -> "

        # === Case 2: LLM Matching ===
        elif self.config.agent_config.use_llm and len(user_input) >= self.config.agent_config.llm_threshold:
            # LLM only takes chat history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            # Keep the last N rounds of chat history specified by the client
            chat_history = messages[-max(1, self.config.agent_config.llm_context_rounds * 2):]
            type_id, type_name, input_summary, inference_type = self.llm_matcher.llm_infer(chat_history)

            if type_name != "其他":
                if inference_type == "意图库":
                    branch_id, branch_type = self.branch_id_lookup.get(type_id, "others"), self.branch_type_lookup.get(type_id, "其他")
                    next_state = branch_id
                    log_info = (
                        f"大模型意图命中 - 意图ID：{type_id} - 意图名称：{type_name} "
                        f"- 意图分支ID：- {branch_id} - 意图分支名称：{branch_type}"
                        f"- 大模型理解：“{input_summary}” "
                        f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                        f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                        f"- 耗时：{(time.time() - prev_time):.3f}秒"
                    )
                    user_logic_title = {f"主线流程【{branch_type}】分支 “{type_name}”", f"大模型理解：“{input_summary}”"}
                    assistant_logic_title = f"【主线流程】{self.config.main_flow_name} 、{branch_type} -> "
                elif inference_type == "知识库":  # If there is no active intentions
                    next_state = "others" # Need to work on this logic
                    inference_use_type = self.knowledge_type_lookup.get(type_id, "无")
                    log_info = (
                        f"大模型知识库意图命中 - 知识库意图ID：{type_id} - 知识库意图名称：{type_name} "
                        f"- 大模型理解：“{input_summary}” "
                        f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                        f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                        f"- 耗时：{(time.time() - prev_time):.3f}秒"
                    )
                    user_logic_title = {f"知识库意图 “{type_name}”", f"大模型理解：“{input_summary}”"}
                    assistant_logic_title = f"【主线流程】{self.config.main_flow_name} -> "
                else:
                    e_m = "意图没有来自意图库或知识库"
                    logger_chatflow.error(e_m)
                    raise ValueError(e_m)
            else:
                log_info = (
                    f"没有意图命中(大模型开启) "
                    f"- 大模型理解：“{input_summary}” "
                    f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                    f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                    f"- 耗时：{(time.time() - prev_time):.3f}秒"
                )
                user_logic_title = {f"没有意图命中(大模型开启)", f"大模型理解：“{input_summary}”"},
                assistant_logic_title = f"【主线流程】{self.config.main_flow_name} -> "
        else:
            # === Case 3: Keyword Matching ===
            type_id, type_name, keywords, count, inference_type = integrated_keywords_matcher(
                user_input,
                self.config.agent_config.intention_priority,
                self.keyword_matcher,
                self.knowledge_keyword_matcher
            )
            if inference_type == "意图库":
                branch_id, branch_type = self.branch_id_lookup.get(type_id, "others"), self.branch_type_lookup.get(type_id, "其他")
                next_state = branch_id
                log_info = (
                    f"关键词意图命中 - 意图ID：{type_id} - 意图名称：{type_name}"
                    f" - {count}个关键词：{'、'.join(keywords)} "
                    f"- 意图分支ID：- {branch_id} - 意图分支名称：{branch_type}"
                    f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                    f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                    f"- 耗时：{(time.time() - prev_time):.3f}秒"
                )
                user_logic_title = {f"主线流程【{branch_type}】分支 “{type_name}”", f"命中{count}个关键词：{'、'.join(keywords)}"}
                assistant_logic_title = f"【主线流程】{self.config.main_flow_name} 、{branch_type} -> "
            elif inference_type == "知识库":
                next_state = "others" #need to work on this logic
                inference_use_type = self.knowledge_type_lookup.get(type_id, "无")
                log_info = (
                    f"关键词知识库意图命中 - 知识库意图ID：{type_id} - 知识库意图名称：{type_name} "
                    f" - {count}个关键词：{'、'.join(keywords)} "
                    f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                    f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                    f"- 耗时：{(time.time() - prev_time):.3f}秒"
                )
                user_logic_title = {f"知识库意图 “{type_name}”"}
                assistant_logic_title = f"【主线流程】{self.config.main_flow_name} -> "
            # === Case 4: Semantic Matching ===
            else:
                type_id, type_name, content, cos_score, inference_type = integrated_semantic_matcher(
                    user_input,
                    self.config.agent_config.cosine_threshold,
                    self.config.agent_config.intention_priority,
                    self.semantic_matcher,
                    self.knowledge_semantic_matcher
                )
                if inference_type == "意图库":
                    branch_id, branch_type = self.branch_id_lookup.get(type_id, "others"), self.branch_type_lookup.get(type_id, "其他")
                    next_state = branch_id
                    log_info = (
                        f"问法意图命中 - 意图ID：{type_id} - 意图名称：{type_name} "
                        f"- 命中问法：{content} - 相似度：{cos_score:.3f} "
                        f"- 意图分支ID：- {branch_id} - 意图分支名称：{branch_type}"
                        f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                        f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                        f"- 耗时：{(time.time() - prev_time):.3f}秒"
                    )
                    user_logic_title = {f"主线流程【{branch_type}】分支 “{type_name}”", f"命中问法：{content} - 相似度：{cos_score:.3f}"}
                    assistant_logic_title = f"【主线流程】{self.config.main_flow_name} 、{branch_type} -> "
                elif inference_type == "知识库":
                    next_state = "others" #need to work on this logic
                    inference_use_type = self.knowledge_type_lookup.get(type_id, "无")
                    log_info = (
                        f"问法知识库意图命中 - 知识库意图ID：{type_id} - 知识库意图名称：{type_name} "
                        f"- 命中问法：{content} - 相似度：{cos_score:.3f} "
                        f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                        f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                        f"- 耗时：{(time.time() - prev_time):.3f}秒"
                    )
                    user_logic_title = {f"知识库意图 “{type_name}”"}
                    assistant_logic_title = f"【主线流程】{self.config.main_flow_name} -> "
                else:
                    log_info = (
                        f"没有意图命中(大模型关闭) "
                        f"- 主流程ID：{self.config.main_flow_id} - 主流程名称：{self.config.main_flow_name} "
                        f"- 节点ID：{self.config.node_id} - 节点名称：{self.config.node_name} "
                        f"- 耗时：{(time.time() - prev_time):.3f}秒"
                    )
                    user_logic_title = {f"没有意图命中(大模型关闭)"}
                    assistant_logic_title = f"【主线流程】{self.config.main_flow_name} -> "

        # Set up metadata
        if branch_type:
            branch_count[branch_type] = branch_count.get(branch_type, 0) + 1
        updated_metadata = state["metadata"] + [
            {
                **previous_metadata,
                "role": "user",
                "content": user_input,
                "intention_tag": type_id,
                "branch_count": branch_count,
                "logic":{
                    "user_logic_title": user_logic_title,
                    "assistant_logic_title": assistant_logic_title,
                    "detail": {
                        "main_flow_id": self.config.main_flow_id,
                        "node_id": self.config.node_id,
                        "branch_id": branch_id,
                        "infer_type": inference_type,
                        "infer_use_type": inference_use_type
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