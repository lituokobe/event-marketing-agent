from typing import Any, Literal
from pydantic import create_model, BaseModel, Field

# Keyword approach
import ahocorasick

# Semantic approach
from pymilvus import MilvusClient

from functionals.log_utils import logger_chatflow
from models.embedding_functions import embed_query

# Models
# from models.embeddings import qwen3_embedding_model
from models.models import qwen_llm

# Configuration
from config.config_setup import NodeConfig

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

# TODO: Create a keyword matching class
# Below method is using ahocorasick, which provide perfect isolation and faster speed.
class KeywordMatcher:
    def __init__(self, intentions: list):
        self.intentions = intentions
        self.keyword_to_id_and_type = {}
        self.all_keywords = set()
        self.automaton = None
        if intentions:
            self.load_keywords_from_dict(intentions)

    def load_keywords_from_dict(self, intentions:list):
        """
        Add all the keywords configured by clients to AC machine.
        """
        for intention in intentions:
            self.add_keyword_list(intention["intention_id"], intention["intention_name"], intention["keywords"])
        self._build_automaton()

    def add_keyword_list(self, intention_id: str, intention_name: str, keywords: list[str] | None):
        if keywords:
            for keyword in keywords:
                keyword = keyword.strip()
                if keyword and keyword not in self.all_keywords:
                    self.keyword_to_id_and_type[keyword] = (intention_id, intention_name)
                    self.all_keywords.add(keyword)

    def _build_automaton(self):
        A = ahocorasick.Automaton()
        for keyword in self.all_keywords:
            A.add_word(keyword, keyword)
        A.make_automaton()
        self.automaton = A

    def analyze_sentence(self, sentence: str) -> dict[str, dict[str, Any]]:
        result = {}
        if self.all_keywords: #
            # Find all matches
            for end_index, keyword in self.automaton.iter(sentence):
                type_id, keyword_type = self.keyword_to_id_and_type[keyword]
                if type_id not in result:
                    result[type_id] = {
                        "keyword_type": keyword_type,
                        "count": 0,
                        "keywords": []
                    }
                result[type_id]["count"] += 1
                result[type_id]["keywords"].append(keyword)
        return result

    @staticmethod
    def get_primary_type(result: dict[str|None, dict[str, Any]|None]) -> tuple[str, str, list[str], int]:
        if not result:
            e_m = "输入的关键词查询结果为空"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)

        primary_id = max(result.keys(), key=lambda k: result[k]['count'])
        info = result[primary_id]
        return primary_id, info["keyword_type"], info["keywords"], info["count"]


# TODO: Create a semantic matching class
class SemanticMatcher:
    def __init__(self, collection_name: str, intention_ids: list, milvus_client: MilvusClient | None = None):
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
        self.config = config
        self.intention_infer_id = {} # dict to store intention inference_type -> type_id
        self.intention_infer_descriptions = {} # dict to store intention inference_type -> description
        for intention in intentions:
            self.intention_infer_id[intention["intention_name"]] = intention["intention_id"]
            self.intention_infer_descriptions[intention["intention_name"]] = " ".join(intention["llm_description"]) if intention["llm_description"] else ""

        # remove the knowledge item that user specifically ask not to include via "nomatch_knowledge_ids"
        nomatch_knowledge_ids = config.other_config.get("nomatch_knowledge_ids", [])
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
        return qwen_llm

    def _create_inference_llm(self, intention_priority: int):
        """
        Create an LLM with specified inference output.
        """
        # Prepare the prompt of the LLM
        docstring_base = [
            self.llm_role_description, self.llm_background_info,
            "你需要输出3个值：",
            "1. 根据对话历史清晰地描述总结最后一次用户输入，然后输出为 input_summary，不超过10个字。",
            "",
            "2. 根据用户的输入来判断意图，然后输出为 user_intention。",
            "用户的意图只能是以下**意图库列表**或**知识库列表**中给定的值。",
            "**意图库列表** （- 意图：意图说明）"
        ]
        # Include the intention descriptions
        for k, v in self.intention_infer_descriptions.items():
            docstring_base.append(f"  - {k}：{v}")

        docstring_base.append("**知识库列表** （- 意图：意图说明）")
        if self.knowledge_infer_description:
            for k, v in self.knowledge_infer_description.items():
                docstring_base.append(f"  - {k}：{v}")

        priority_map = [
            "", # 1知识库优先 2回答分支优先 3只能匹配优先
            "请牢记两个列表的使用顺序：首先使用**知识库列表**判别用户的意图，并将意图输出为 user_intention。如果无法判别，则使用**意图库列表**判别用户的意图，并将意图输出为 user_intention",
            "请牢记两个列表的使用顺序：首先使用**意图库列表**判别用户的意图，并将意图输出为 user_intention。如果无法判别，则使用**知识库列表**判别用户的意图，并将意图输出为 user_intention",
            "同时使用**知识库列表**和**意图库列表**判别用户的意图，不分先后，并将意图输出为 user_intention"
        ]

        docstring_tail = [
            "如果用户的意图不属于两个列表中的任何意图，或者用户没有任何输入，输出 user_intention 为 '其他'。",
            "请一定要调用这个工具，并输出指定值。",
            "你只能输出最合适的唯一值，不能输出两个或多个。",
            "",
            "3. 将判断意图的所使用的列表输出为 inference_type。",
            "如果是使用**意图库列表**判断，输出 inference_type 为'意图库'；",
            "如果是使用**知识库判断**判断，输出 inference_type 为'知识库'；",
            "如果没有判断出意图，即用户的意图不属于意图库和知识库列表中的任何意图，或者用户没有任何输入，输出 inference_type 为'无'。",
        ]

        docstring = "\n".join(docstring_base + [priority_map[intention_priority]] + docstring_tail)
        # print(f"当前节点：{self.config.node_id}-{self.config.node_name}\n 以下是大模型提示词 \n {docstring}")
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

    def llm_infer(self, chat_history: list[dict|None]) -> tuple[str, str, str, str]:
        """
        Infer the user intention from the user input.
        """
        if not self.llm_runnable:
            e_m = "LLM推理工具未初始化"
            logger_chatflow.error(e_m)
            raise RuntimeError(e_m)

        user_intention, input_summary, inference_type = "其他", "无", "无"

        try:
            resp = self.llm_runnable.invoke(chat_history)
            if resp.tool_calls:
                args = resp.tool_calls[0]["args"]
                user_intention = args.get("user_intention", "其他")
                input_summary = args.get("input_summary", "无")
                inference_type = args.get("inference_type", "无")
        except Exception as e:
            logger_chatflow.error("LLM推理调用异常：%s", {e})

        if inference_type == "意图库":
            return self.intention_infer_id.get(user_intention, "others"), user_intention, input_summary, inference_type
        elif inference_type == "知识库":
            return self.knowledge_infer_id.get(user_intention, "others"), user_intention, input_summary, inference_type
        else:
            return "others", user_intention, input_summary, inference_type