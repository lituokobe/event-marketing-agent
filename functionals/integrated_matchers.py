from functionals.log_utils import logger_chatflow
from functionals.matchers import KeywordMatcher, SemanticMatcher


# Combine the intention keyword matcher and the knowledge keyword matcher based on user's preference of intention_priority
def integrated_keywords_matcher(user_input:str,
                                intention_priority:int,
                                keyword_matcher:KeywordMatcher,
                                knowledge_keyword_matcher:KeywordMatcher):
    """
    Infer user intention based on integrated keyword matching.
    Priority can be '回答分支优先', '知识库优先', or '智能匹配优先'.
    Returns: (type_id, type_name, keywords, count, inference_type)
    """
    type_id, type_name, keywords, count, inference_type = "", "", [], 0, "无"

    # Define a function to match the user input with inference type output
    def try_match(matcher: KeywordMatcher, inference_label: str):
        result = matcher.analyze_sentence(user_input)
        if result:
            return (*matcher.get_primary_type(result), inference_label)
        return None

    if intention_priority == 2:
        match = try_match(keyword_matcher, "意图库") or try_match(knowledge_keyword_matcher, "知识库")
    elif intention_priority == 1:
        match = try_match(knowledge_keyword_matcher, "知识库") or try_match(keyword_matcher, "意图库")
    elif intention_priority == 3:
        intention_result = keyword_matcher.analyze_sentence(user_input)
        knowledge_result = knowledge_keyword_matcher.analyze_sentence(user_input)
        integrated_result = intention_result | knowledge_result if intention_result or knowledge_result else {}

        if integrated_result:
            type_id, type_name, keywords, count = keyword_matcher.get_primary_type(integrated_result)
            inference_type = "知识库" if type_id in knowledge_result else "意图库"
            return type_id, type_name, keywords, count, inference_type
        match = None
    else:# If there is no active intentions
        e_m = "优先选择仅能为‘知识库优先’，‘回答分支优先’或‘智能匹配优先’"
        logger_chatflow.error(e_m)
        raise ValueError(e_m)
    if match:
        type_id, type_name, keywords, count, inference_type = match
    return type_id, type_name, keywords, count, inference_type

# Combine the intention semantic matcher and the knowledge semantic matcher based on user's preference of intention_priority
def integrated_semantic_matcher(user_input:str,
                                nlp_threshold:float,
                                intention_priority: int,
                                semantic_matcher:SemanticMatcher,
                                knowledge_semantic_matcher:SemanticMatcher):
    """
    Infer user intention using semantic similarity.
    Priority can be '回答分支优先', '知识库优先', or '智能匹配优先'.
    Returns: (type_id, type_name, content, cos_score, inference_type)
    """
    type_id, type_name, content, cos_score, inference_type = "", "", "", 0.0, "无"

    def try_match(matcher: SemanticMatcher, label: str):
        result = matcher.find_most_similar(user_input)
        if result:
            tid, tname, cont, score = result
            if score > nlp_threshold:
                return tid, tname, cont, score, label
        return None

    if intention_priority == 2:
        match = try_match(semantic_matcher, "意图库") or try_match(knowledge_semantic_matcher, "知识库")
    elif intention_priority == 1:
        match = try_match(knowledge_semantic_matcher, "知识库") or try_match(semantic_matcher, "意图库")
    elif intention_priority == 3:  # If there is no active intentions
        intention_result = semantic_matcher.find_most_similar(user_input)
        knowledge_result = knowledge_semantic_matcher.find_most_similar(user_input)
        # Unpack safely
        if intention_result and knowledge_result:
            tid_i, tname_i, cont_i, score_i = intention_result
            tid_k, tname_k, cont_k, score_k = knowledge_result

            if score_i > nlp_threshold and score_i >= score_k:
                match = (tid_i, tname_i, cont_i, score_i, "意图库")
            elif score_k > nlp_threshold and score_k > score_i:
                match = (tid_k, tname_k, cont_k, score_k, "知识库")
            else:
                match = None
        else:
            match = None
    else:
        e_m = "优先选择仅能为‘知识库优先’，‘回答分支优先’或‘智能匹配优先’"
        logger_chatflow.error(e_m)
        raise ValueError(e_m)
    if match:
        type_id, type_name, content, cos_score, inference_type = match
    return type_id, type_name, content, cos_score, inference_type