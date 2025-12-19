import copy
from typing import Any, Literal
from pydantic import create_model, BaseModel, Field

# Keyword approach
import re
import ahocorasick

# Semantic approach
from pymilvus import MilvusClient

from functionals.log_utils import logger_chatflow
from functionals.embedding_functions import embed_query

# Models
# from models.embeddings import qwen3_embedding_model
from models.models import qwen_llm, deepseek_llm, glm_llm

# Configuration
from config.config_setup import NodeConfig

# String assets
from data.string_asset import docstring_base_raw, priority_map, docstring_tail

"""
3 type of matchers:
KeywordMatcher, based on keyword detection to have user's intention
SemanticMatcher, based on vector semantic matching to have user's intention
LLMInferenceMatcher, based on LLM's inference to have user's intention

KeywordMatcher.get_primary_type(), SemanticMatcher.find_most_similar(), LLMInferenceMatcher.llm_infer()
will all return 5 values:
- user's intention id
- user's intention type
- extra info 1
- extra info 2
- inference type: 意图库 or 知识库
"""

# TODO: Create a keyword matching class, supporting regular expression
# Below method is using ahocorasick, which provide perfect isolation and faster speed.
class KeywordMatcher:
    def __init__(self, intentions: list):
        self.intentions = intentions
        self.keyword_to_id_and_type = {}
        self.all_keywords = set()
        self.regex_patterns = []  # List of tuples: (compiled_regex, original_pattern, intention_id, intention_name)
        self.automaton = None
        if intentions:
            self.load_keywords_from_dict(intentions)

    def _is_probably_regex(self, pattern: str) -> bool:
        """
        Heuristic to detect if a string is intended as a regex.
        You can adjust this logic if needed (e.g., require explicit flag).
        """
        regex_meta = {'^', '$', '|', '(', ')', '*', '+', '?', '[', '{', '\\'}
        return any(char in regex_meta for char in pattern)

    def load_keywords_from_dict(self, intentions: list):
        self.keyword_to_id_and_type.clear()
        self.all_keywords.clear()
        self.regex_patterns.clear()

        for intention in intentions:
            self.add_keyword_list(
                intention["intention_id"],
                intention["intention_name"],
                intention["keywords"]
            )
        self._build_automaton()

    def add_keyword_list(self, intention_id: str, intention_name: str, keywords: list[str] | None):
        if not keywords:
            return
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue
            if self._is_probably_regex(keyword):
                # Store compiled regex + metadata
                try:
                    compiled = re.compile(keyword)
                    self.regex_patterns.append((compiled, keyword, intention_id, intention_name))
                except re.error:
                    # Optionally log or skip invalid regex
                    continue
            else:
                if keyword not in self.all_keywords:
                    self.keyword_to_id_and_type[keyword] = (intention_id, intention_name)
                    self.all_keywords.add(keyword)

    def _build_automaton(self):
        if not self.all_keywords:
            self.automaton = None
            return
        A = ahocorasick.Automaton()
        for keyword in self.all_keywords:
            A.add_word(keyword, keyword)
        A.make_automaton()
        self.automaton = A

    def analyze_sentence(self, sentence: str) -> dict[str, dict[str, Any]]:
        result = {}

        # 1. Match literal keywords using Aho-Corasick
        if self.automaton:
            for end_index, keyword in self.automaton.iter(sentence):
                intention_id, keyword_type = self.keyword_to_id_and_type[keyword]
                if intention_id not in result:
                    result[intention_id] = {
                        "keyword_type": keyword_type,
                        "count": 0,
                        "keywords": []
                    }
                result[intention_id]["count"] += 1
                result[intention_id]["keywords"].append(keyword)

        # 2. Match regex patterns
        for compiled_regex, original_pattern, intention_id, keyword_type in self.regex_patterns:
            # Use finditer to find all non-overlapping matches
            matches = list(compiled_regex.finditer(sentence))
            if matches:
                if intention_id not in result:
                    result[intention_id] = {
                        "keyword_type": keyword_type,
                        "count": 0,
                        "keywords": []
                    }
                count = len(matches)
                result[intention_id]["count"] += count
                # Append the original regex pattern once per match (as requested)
                result[intention_id]["keywords"].extend([original_pattern] * count)
        return result

    @staticmethod
    def get_primary_type(result: dict[str | None, dict[str, Any] | None]) -> tuple[str, str, list[str], int]:
        if not result:
            e_m = "输入的关键词查询结果为空"
            logger_chatflow.error(e_m)
            return "others", "", [], 0

        primary_id = max(result.keys(), key=lambda k: result[k]['count'])
        info = result[primary_id]
        return primary_id, info["keyword_type"], info["keywords"], info["count"]

# TODO: Create a semantic matching class
class SemanticMatcher:
    def __init__(self, collection_name: str, intention_ids: set|list, milvus_client: MilvusClient | None = None):
        self.collection_name = collection_name
        self.milvus_client = milvus_client
        self.intention_ids = intention_ids

    def find_most_similar(self, sentence: str) -> tuple[str, str, str, float]:
        """
        Find the most similar intention to the given sentence.

        Returns:
            tuple: (intention_id, intention_name, phrase, similarity_score)
            Always returns a valid tuple even when no match is found
        """
        DEFAULT_RESULT = ("无", "无", "无", 0.0)
        if not self.intention_ids:
            return DEFAULT_RESULT

        try:
            # Generate query embedding
            # query_emb = qwen3_embedding_model.embed_documents(sentence)
            # User embed_documents from embedding_service-embedding_model
            query_emb = embed_query(sentence)
            if hasattr(query_emb, 'tolist'):
                query_emb = query_emb.tolist()

            # Build filter expression: intention_id in ["K003", "I008", ...], Milvus uses string expressions
            id_list_str = ",".join(f'"{id_}"' for id_ in self.intention_ids)
            filter_expr = f"intention_id in [{id_list_str}]"

            # Perform search
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_emb],
                filter=filter_expr,
                limit=1,
                output_fields=["intention_id", "intention_name", "phrase"],
                time_out = 3.0
            )

            # Check if we have results
            if not results or not results[0]:
                return DEFAULT_RESULT

            # Extract result safely
            hit = results[0][0] # hit is a Milvus hit object, similar to dict
            entity = hit.get("entity", {})

            return (entity.get("intention_id", "无"), entity.get("intention_name", "无"),
                entity.get("phrase", "无"), hit.get("distance", 0.0))

        except Exception as e:
            logger_chatflow.error(f"'{sentence}'查询失败: {str(e)}", exc_info=True)
            # Final fallback
            return DEFAULT_RESULT

# TODO: Create a LLM inference matching class
class LLMInferenceMatcher:
    def __init__(self,
                 config: NodeConfig,
                 intentions:list[dict],
                 knowledge_infer_id: dict,
                 knowledge_infer_description: dict,
                 intention_priority: int):
        self.config = copy.deepcopy(config)
        self.intention_infer_id = {} # dict to store intention inference_type -> type_id
        self.intention_infer_descriptions = {} # dict to store intention inference_type -> description
        for intention in intentions:
            self.intention_infer_id[intention["intention_name"]] = intention["intention_id"]
            self.intention_infer_descriptions[intention["intention_name"]] = " ".join(intention["llm_description"]) if intention["llm_description"] else ""

        # remove the knowledge item that user specifically ask not to include via "nomatch_knowledge_ids"
        nomatch_knowledge_ids = self.config.other_config.get("nomatch_knowledge_ids", [])
        if not isinstance(nomatch_knowledge_ids, list):
            e_m = f"{config.node_id}-{config.node_name}节点nomatch_knowledge_ids应为列表"
            logger_chatflow.error(e_m)
            raise TypeError(e_m)
        self.knowledge_infer_id = {k:v for k, v in knowledge_infer_id.items() if v not in nomatch_knowledge_ids} # dict to store knowledge inference_type -> type_id
        self.knowledge_infer_description = {k:v for k, v in knowledge_infer_description.items() if k in self.knowledge_infer_id} # dict to store knowledge inference_type -> description

        # background prompt
        self.llm_role_description = getattr(config.agent_config, "llm_role_description", "")
        self.llm_background_info = getattr(config.agent_config, "llm_background_info", "")

        # select llm runnable
        self.llm_runnable = self._select_llm(config.agent_config.llm_name)

        # bind inference model
        self.llm_runnable = self._create_inference_llm(intention_priority)

    def _select_llm(self, llm_name: str):
        if llm_name == "qwen_llm":
            return qwen_llm
        elif llm_name == "deepseek_llm":
            return deepseek_llm
        elif llm_name == "glm_llm":
            return glm_llm
        return deepseek_llm

    def _create_inference_llm(self, intention_priority: int):
        """
        Create an LLM with specified inference output.
        """
        #TODO: Prepare the prompt of the LLM
        #Add the role description and background info
        docstring_base = [self.llm_role_description, self.llm_background_info] + docstring_base_raw

        #Add intention priority
        docstring_base = docstring_base + [priority_map[intention_priority]] + priority_map[4:]

        # Include the intention descriptions
        docstring_base.append("**【意图库列表】**（- 意图：意图说明）")
        if self.intention_infer_descriptions:
            for k, v in self.intention_infer_descriptions.items():
                docstring_base.append(f"  - {k}：{v}")
        else:
            docstring_base.append("")

        docstring_base.append("**【知识库列表】**（- 意图：意图说明）")
        if self.knowledge_infer_description: # if knowledge exists
            for k, v in self.knowledge_infer_description.items():
                docstring_base.append(f"  - {k}：{v}")
        else: # if knowledge doesn't exist, append an empty string as a blank row later
            docstring_base.append("")

        # Combine all the strings in to docstring as prompt
        docstring = "\n".join(docstring_base + docstring_tail)
        # print(f"{self.config.node_id}-{self.config.node_name}节点的大模型提示词 \n{docstring}")
        # print()
        # Assemble the prompt based user's preference of intention priority
        all_intentions = list(self.intention_infer_id.keys()) + list(self.knowledge_infer_id.keys()) + ["其他"]

        InferenceModel = create_model(
            "InferenceModel",
            __base__=BaseModel,
            input_summary=(str, Field(description="描述最后一次用户输入")),
            user_intention=(Literal[*all_intentions], Field(description="判断用户意图的结果")),
            inference_type=(Literal["意图库", "知识库", "无"], Field(description="判断用户意图的依据")),
            __doc__=docstring
        )

        return self.llm_runnable.bind_tools([InferenceModel])

    def llm_infer(self, chat_history: list[dict|None]) -> tuple[str, str, str, str, int]:
        """
        Infer the user intention from the user input.
        """
        if not self.llm_runnable:
            e_m = "LLM推理工具未初始化"
            logger_chatflow.error(e_m)
            raise RuntimeError(e_m)

        # Initialize default values
        user_intention, input_summary, inference_type, token_used = "其他", "无", "无", 0

        try:
            resp = self.llm_runnable.invoke(chat_history)
            # Get the tokens consumed per round of conversation including the preconfigured doc string, full chat history, AI reply, etc.
            token_used = int(resp.response_metadata.get("token_usage", {}).get("total_tokens", 0))
            if resp.tool_calls:
                print("大模型调用工具")
                args = resp.tool_calls[0]["args"]
                user_intention = args.get("user_intention", "其他")
                input_summary = args.get("input_summary", "无")
                inference_type = args.get("inference_type", "无")
                print(f"user_intention: {user_intention}, inference_type: {inference_type}, "
                      f"input_summary: {input_summary}")
            else:
                print("大模型没有调用工具")
        except Exception as e:
            logger_chatflow.error("LLM推理调用异常：%s", {e})

        if inference_type == "意图库":
            return (self.intention_infer_id.get(user_intention, "others"), user_intention,
                    input_summary, inference_type, token_used)
        elif inference_type == "知识库":
            return (self.knowledge_infer_id.get(user_intention, "others"), user_intention,
                    input_summary, inference_type, token_used)
        else:
            return "others", user_intention, input_summary, inference_type, token_used