import copy
import time
from pymilvus import MilvusClient
from config.config_setup import NodeConfig, KnowledgeContext, GlobalConfigContext
from data.string_asset import infer_tool_str
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
                 global_config_context:GlobalConfigContext,
                 intentions: list,
                 milvus_client: MilvusClient | None = None,
                 ):
        self.config = config
        self.knowledge_type_lookup = knowledge_context.type_lookup
        self.knowledge_match_lookup = knowledge_context.match_lookup
        self.global_no_input = global_config_context.no_input
        self.global_no_infer_result = global_config_context.no_infer_result

        # Get active intention ids to filter the intention
        # Get active intention id - intention branch id lookup table
        self.branch_id_lookup, self.branch_name_lookup, self.branch_type_lookup, active_intention_ids = {}, {}, {}, []

        # create active_intention_ids
        # create lookup table to look for branch_id, branch_name, and branch_type, with intention_id
        for branch in config.intention_branches:
            intention_ids = branch.get("intention_ids", [])
            if isinstance(intention_ids, list):
                for intention_id in intention_ids:
                    if intention_id in self.branch_id_lookup:
                        e_m = f"意图不能重复"
                        logger_chatflow.error(e_m)
                        raise ValueError(e_m)
                    active_intention_ids.append(intention_id)
                    self.branch_id_lookup[intention_id] = branch.get("branch_id")
                    self.branch_name_lookup[intention_id] = branch.get("branch_name")
                    self.branch_type_lookup[intention_id] = branch.get("branch_type")

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
        nomatch_knowledge_ids = config.other_config.get("nomatch_knowledge_ids", [])
        if not isinstance(nomatch_knowledge_ids, list):
            e_m = f"{config.node_id}-{config.node_name}节点nomatch_knowledge_ids应为列表"
            logger_chatflow.error(e_m)
            raise TypeError(e_m)
        """
        When use_llm is off or 
        llm_threshold > 0 even if use_llm is on (meaning we still use the traditional approaches if user input is below this threshold),
        We initialize the matchers of these traditional approaches: keyword and semantic
        """
        if config.agent_config.enable_nlp == 1: # Use semantic matching globally
            if config.agent_config.use_llm != 1 or config.agent_config.llm_threshold > 0:
                # for intentions specified in this node
                self.keyword_matcher = KeywordMatcher(filtered_intentions)
                self.semantic_matcher = SemanticMatcher(
                    config.agent_config.collection_name,
                    active_intention_ids, # No need the full intention content, just the ids
                    milvus_client
                )
                # for intentions from knowledge
                if nomatch_knowledge_ids:
                    knowledge_without_nomatch = []
                    knowledge_ids_without_nomatch = []
                    for item in knowledge_context.knowledge:
                        if item["intention_id"] not in nomatch_knowledge_ids:
                            knowledge_without_nomatch.append(item)
                            knowledge_ids_without_nomatch.append(item["intention_id"])

                    self.knowledge_keyword_matcher = KeywordMatcher(knowledge_without_nomatch)
                    self.knowledge_semantic_matcher = SemanticMatcher(
                        config.agent_config.collection_name,
                        knowledge_ids_without_nomatch,
                        milvus_client
                    )
                else:
                    self.knowledge_keyword_matcher = knowledge_context.keyword_matcher
                    self.knowledge_semantic_matcher = knowledge_context.semantic_matcher
        else:
            if config.agent_config.use_llm != 1 or config.agent_config.llm_threshold > 0:
                # for intentions specified in this node
                self.keyword_matcher = KeywordMatcher(filtered_intentions)
                # for intentions from knowledge
                if nomatch_knowledge_ids:
                    knowledge_without_nomatch = [item for item in knowledge_context.knowledge if item["intention_id"] not in nomatch_knowledge_ids]
                    self.knowledge_keyword_matcher = KeywordMatcher(knowledge_without_nomatch)
                else:
                    self.knowledge_keyword_matcher = knowledge_context.keyword_matcher

        if config.agent_config.use_llm == 1:
            self.llm_matcher = LLMInferenceMatcher(config, # include the "nomatch_knowledge_ids" argument
                                                   filtered_intentions,
                                                   knowledge_context.infer_id,
                                                   knowledge_context.infer_description,
                                                   config.agent_config.intention_priority)

    def __call__(self, state: ChatState) -> dict:
        # Get the message and last user message
        messages = state["messages"]
        user_input = get_last_user_message(messages).lower()
        print(f"{self.config.node_id}_intention-{self.config.node_name}节点开始工作")
        
        # Get the last log info
        logs = state.get("logs", [])
        previous_log = logs[-1] if logs else {}
        branch_type_count = copy.deepcopy(previous_log.get("branch_type_count", {}))
        knowledge_match_balance = copy.deepcopy(previous_log.get("knowledge_match_balance", copy.deepcopy(self.knowledge_match_lookup))) # balance knowledge item matches, use deepcopy to isolate each instance
        
        # Record the time
        prev_time = time.time()

        # Set up default outputs
        next_state = "others"

        # Identify user intention
        # === Case 1: Empty input ===
        if not user_input:
            if self.global_no_input:
                next_state = "no_input"
            log_info = {
                "role": "user",
                "content": user_input,
                "main_flow_id": self.config.main_flow_id,
                "main_flow_name": self.config.main_flow_name,
                "node_id": self.config.node_id,
                "node_name": self.config.node_name,
                "match_to": "没有意图命中",
                "branch_id": "",
                "branch_name": "",
                "branch_type": "",
                "branch_type_count": branch_type_count,
                "intention_id": "",
                "intention_name": "",
                "infer_tool": infer_tool_str[0],
                "llm_input_summary": "",
                "matching_content": "",
                "matching_score": 0.0,
                "knowledge_type": "",
                "knowledge_match_balance": knowledge_match_balance,
                "other_config": self.config.other_config,
                "time_cost": 0.0
            }
        # === Case 2: LLM Matching ===
        elif self.config.agent_config.use_llm == 1 and len(user_input) >= self.config.agent_config.llm_threshold:
            # LLM only takes chat history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            # Keep the last N rounds of chat history specified by the client
            chat_history = messages[-max(1, self.config.agent_config.llm_context_rounds * 2):]
            type_id, type_name, input_summary, infer_type = self.llm_matcher.llm_infer(chat_history)

            if type_name != "其他":
                if infer_type == "意图库":
                    branch_id, branch_name, branch_type = (self.branch_id_lookup.get(type_id, "others"),
                                                           self.branch_name_lookup.get(type_id, "其他"),
                                                           self.branch_type_lookup.get(type_id, "others"))
                    branch_type_count[branch_type] = branch_type_count.get(branch_type, 0) + 1
                    next_state = branch_id
                    log_info = {
                        "role": "user",
                        "content": user_input,
                        "main_flow_id": self.config.main_flow_id,
                        "main_flow_name": self.config.main_flow_name,
                        "node_id": self.config.node_id,
                        "node_name": self.config.node_name,
                        "match_to": "主线流程",
                        "branch_id": branch_id,
                        "branch_name": branch_name,
                        "branch_type": branch_type,
                        "branch_type_count": branch_type_count,
                        "intention_id": type_id,
                        "intention_name": type_name,
                        "infer_tool": infer_tool_str[1],
                        "llm_input_summary": input_summary,
                        "matching_content": "",
                        "matching_score": 0.0,
                        "knowledge_type": "",
                        "knowledge_match_balance": knowledge_match_balance,
                        "other_config": self.config.other_config,
                        "time_cost": round(time.time() - prev_time, 3)
                    }
                elif infer_type == "知识库":  # If there is no active intentions
                    knowledge_type = ""
                    # Process according to current match balance
                    if not isinstance(knowledge_match_balance.get(type_id), int):
                        e_m = f"{type_id}不在知识库中"
                        logger_chatflow.error(e_m)
                        raise TypeError(e_m)
                    if knowledge_match_balance[type_id] > 0:
                        next_state = type_id  # Navigate to the knowledge reply sub-node
                        knowledge_match_balance[type_id] -= 1
                        knowledge_type = self.knowledge_type_lookup.get(type_id)
                    elif self.global_no_infer_result:
                        next_state = "no_infer_result"

                    log_info = {
                        "role": "user",
                        "content": user_input,
                        "main_flow_id": self.config.main_flow_id,
                        "main_flow_name": self.config.main_flow_name,
                        "node_id": self.config.node_id,
                        "node_name": self.config.node_name,
                        "match_to": "知识库",
                        "branch_id": "",
                        "branch_name": "",
                        "branch_type": "",
                        "branch_type": "",
                        "branch_type_count": branch_type_count,
                        "intention_id": type_id,
                        "intention_name": type_name,
                        "infer_tool": infer_tool_str[1],
                        "llm_input_summary": input_summary,
                        "matching_content": "",
                        "matching_score": 0.0,
                        "knowledge_type": knowledge_type,
                        "knowledge_match_balance": knowledge_match_balance,
                        "other_config": self.config.other_config,
                        "time_cost": round(time.time() - prev_time, 3)
                    }
                else:
                    e_m = "意图没有来自意图库或知识库"
                    logger_chatflow.error(e_m)
                    raise ValueError(e_m)
            else:
                if self.global_no_infer_result:
                    next_state = "no_infer_result"
                log_info = {
                    "role": "user",
                    "content": user_input,
                    "main_flow_id": self.config.main_flow_id,
                    "main_flow_name": self.config.main_flow_name,
                    "node_id": self.config.node_id,
                    "node_name": self.config.node_name,
                    "match_to": "没有意图命中",
                    "branch_id": "",
                    "branch_name": "",
                    "branch_type": "",
                    "branch_type_count": branch_type_count,
                    "intention_id": "",
                    "intention_name": "",
                    "infer_tool": infer_tool_str[1],
                    "llm_input_summary": input_summary,
                    "matching_content": "",
                    "matching_score": 0.0,
                    "knowledge_type": "",
                    "knowledge_match_balance": knowledge_match_balance,
                    "other_config": self.config.other_config,
                    "time_cost": round(time.time() - prev_time, 3)
                }
        else:
            # === Case 3: Keyword Matching ===
            type_id, type_name, keywords, count, infer_type = integrated_keywords_matcher(
                user_input,
                self.config.agent_config.intention_priority,
                self.keyword_matcher,
                self.knowledge_keyword_matcher
            )
            if infer_type == "意图库":
                branch_id, branch_name, branch_type = (self.branch_id_lookup.get(type_id, "others"),
                                                       self.branch_name_lookup.get(type_id, "其他"),
                                                       self.branch_type_lookup.get(type_id, "others"))
                branch_type_count[branch_type] = branch_type_count.get(branch_type, 0) + 1
                next_state = branch_id
                log_info = {
                    "role": "user",
                    "content": user_input,
                    "main_flow_id": self.config.main_flow_id,
                    "main_flow_name": self.config.main_flow_name,
                    "node_id": self.config.node_id,
                    "node_name": self.config.node_name,
                    "match_to": "主线流程",
                    "branch_id": branch_id,
                    "branch_name": branch_name,
                    "branch_type": branch_type,
                    "branch_type_count": branch_type_count,
                    "intention_id": type_id,
                    "intention_name": type_name,
                    "infer_tool": infer_tool_str[2],
                    "llm_input_summary": "",
                    "matching_content": "、".join(keywords),
                    "matching_score": float(count),
                    "knowledge_type": "",
                    "knowledge_match_balance": knowledge_match_balance,
                    "other_config": self.config.other_config,
                    "time_cost": round(time.time() - prev_time, 3)
                }
            elif infer_type == "知识库":
                knowledge_type = ""
                # Process according to current match balance
                if not isinstance(knowledge_match_balance.get(type_id), int):
                    e_m = f"{type_id}不在知识库中"
                    logger_chatflow.error(e_m)
                    raise TypeError(e_m)
                if knowledge_match_balance[type_id] > 0:
                    next_state = type_id  # Navigate to the knowledge reply sub-node
                    knowledge_match_balance[type_id] -= 1
                    knowledge_type = self.knowledge_type_lookup.get(type_id)
                elif self.global_no_infer_result:
                    next_state = "no_infer_result"

                log_info = {
                    "role": "user",
                    "content": user_input,
                    "main_flow_id": self.config.main_flow_id,
                    "main_flow_name": self.config.main_flow_name,
                    "node_id": self.config.node_id,
                    "node_name": self.config.node_name,
                    "match_to": "知识库",
                    "branch_id": "",
                    "branch_name": "",
                    "branch_type": "",
                    "branch_type_count": branch_type_count,
                    "intention_id": type_id,
                    "intention_name": type_name,
                    "infer_tool": infer_tool_str[2],
                    "llm_input_summary": "",
                    "matching_content": "、".join(keywords),
                    "matching_score": float(count),
                    "knowledge_type": knowledge_type,
                    "knowledge_match_balance": knowledge_match_balance,
                    "other_config": self.config.other_config,
                    "time_cost": round(time.time() - prev_time, 3)
                }
            # === Case 4: Semantic Matching ===
            elif self.config.agent_config.enable_nlp == 1:
                type_id, type_name, content, cos_score, infer_type = integrated_semantic_matcher(
                    user_input,
                    self.config.agent_config.nlp_threshold,
                    self.config.agent_config.intention_priority,
                    self.semantic_matcher,
                    self.knowledge_semantic_matcher
                )
                if infer_type == "意图库":
                    branch_id, branch_name, branch_type = (self.branch_id_lookup.get(type_id, "others"),
                                                           self.branch_name_lookup.get(type_id, "其他"),
                                                           self.branch_type_lookup.get(type_id, "others"))
                    branch_type_count[branch_type] = branch_type_count.get(branch_type, 0) + 1
                    next_state = branch_id
                    log_info = {
                        "role": "user",
                        "content": user_input,
                        "main_flow_id": self.config.main_flow_id,
                        "main_flow_name": self.config.main_flow_name,
                        "node_id": self.config.node_id,
                        "node_name": self.config.node_name,
                        "match_to": "主线流程",
                        "branch_id": branch_id,
                        "branch_name": branch_name,
                        "branch_type": branch_type,
                        "branch_type_count": branch_type_count,
                        "intention_id": type_id,
                        "intention_name": type_name,
                        "infer_tool": infer_tool_str[3],
                        "llm_input_summary": "",
                        "matching_content": content,
                        "matching_score": round(cos_score, 3),
                        "knowledge_type": "",
                        "knowledge_match_balance": knowledge_match_balance,
                        "other_config": self.config.other_config,
                        "time_cost": round(time.time() - prev_time, 3)
                    }
                elif infer_type == "知识库":
                    knowledge_type = ""
                    # Process according to current match balance
                    if not isinstance(knowledge_match_balance.get(type_id), int):
                        e_m = f"{type_id}不在知识库中"
                        logger_chatflow.error(e_m)
                        raise TypeError(e_m)
                    if knowledge_match_balance[type_id] > 0:
                        next_state = type_id  # Navigate to the knowledge reply sub-node
                        knowledge_match_balance[type_id] -= 1
                        knowledge_type = self.knowledge_type_lookup.get(type_id)
                    elif self.global_no_infer_result:
                        next_state = "no_infer_result"

                    log_info = {
                        "role": "user",
                        "content": user_input,
                        "main_flow_id": self.config.main_flow_id,
                        "main_flow_name": self.config.main_flow_name,
                        "node_id": self.config.node_id,
                        "node_name": self.config.node_name,
                        "match_to": "知识库",
                        "branch_id": "",
                        "branch_name": "",
                        "branch_type": "",
                        "branch_type_count": branch_type_count,
                        "intention_id": type_id,
                        "intention_name": type_name,
                        "infer_tool": infer_tool_str[3],
                        "llm_input_summary": "",
                        "matching_content": content,
                        "matching_score": round(cos_score, 3),
                        "knowledge_type": knowledge_type,
                        "knowledge_match_balance": knowledge_match_balance,
                        "other_config": self.config.other_config,
                        "time_cost": round(time.time() - prev_time, 3)
                    }
                else:
                    if self.global_no_infer_result:
                        next_state = "no_infer_result"
                    log_info = {
                        "role": "user",
                        "content": user_input,
                        "main_flow_id": self.config.main_flow_id,
                        "main_flow_name": self.config.main_flow_name,
                        "node_id": self.config.node_id,
                        "node_name": self.config.node_name,
                        "match_to": "没有意图命中",
                        "branch_id": "",
                        "branch_name": "",
                        "branch_type": "",
                        "branch_type_count": branch_type_count,
                        "intention_id": "",
                        "intention_name": "",
                        "infer_tool": infer_tool_str[4],
                        "llm_input_summary": "",
                        "matching_content": "",
                        "matching_score": 0.0,
                        "knowledge_type": "",
                        "knowledge_match_balance": knowledge_match_balance,
                        "other_config": self.config.other_config,
                        "time_cost": round(time.time() - prev_time, 3)
                    }
            else:
                if self.global_no_infer_result:
                    next_state = "no_infer_result"
                log_info = {
                    "role": "user",
                    "content": user_input,
                    "main_flow_id": self.config.main_flow_id,
                    "main_flow_name": self.config.main_flow_name,
                    "node_id": self.config.node_id,
                    "node_name": self.config.node_name,
                    "match_to": "没有意图命中",
                    "branch_id": "",
                    "branch_name": "",
                    "branch_type": "",
                    "branch_type_count": branch_type_count,
                    "intention_id": "",
                    "intention_name": "",
                    "infer_tool": infer_tool_str[5],
                    "llm_input_summary": "",
                    "matching_content": "",
                    "matching_score": 0.0,
                    "knowledge_type": "",
                    "knowledge_match_balance": knowledge_match_balance,
                    "other_config": self.config.other_config,
                    "time_cost": round(time.time() - prev_time, 3)
                }

        # A correct flow should have a defined next_state at this time
        if next_state == "others":
            e_m = "用户没有输入或意图无法判断。请前往全局配置以应对此种情况。"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)

        updated_logs = logs + [log_info]

        # Log information
        if self.config.enable_logging:
            filtered_logs = [f"{k}:{v}" for k, v in log_info.items() if v]
            if filtered_logs:
                logger_chatflow.info("系统消息：%s", ", ".join(filtered_logs))

        print(f"{self.config.node_id}_intention-{self.config.node_name}节点完成工作")

        return {
            "messages": messages,
            "dialog_state": next_state,
            "logs":updated_logs,
            "metadata": state["metadata"]
        }