# Retrieve the last message from the user in the stack of messages
from functionals.log_utils import logger_chatflow

def get_last_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    e_m = "没有用户输入"
    logger_chatflow.error(e_m)
    raise ValueError(e_m)

# Filter the keywords/semantic dictionary to only include entries with selected ids.
def str_dict_select(str_dict: dict, ids: list[str] | None) -> dict:
    """
    Filter the keywords/semantic dictionary to only include entries with selected ids.
    Args:
        str_dict: Original dict in format {"001": {"label": [...]}, ...}
        ids: List of selected ids (e.g., ["001", "003"])

    Returns:
        Filtered dict containing only the specified str_dicts.
        If there is no ids input, the filtered dict is empty.

    Raises:
        ValueError: If any id is not found in the dict.
    """
    filtered = {}
    missing = []
    if ids:
        for _id in ids:
            if _id in str_dict:
                filtered[_id] = str_dict[_id]
            else:
                missing.append(_id)
        if missing:
            e_m = f"以下意图id不存在：{missing}"
            logger_chatflow.error(e_m)
            raise ValueError(e_m)
    return filtered

# Filter the intention list to only include entries with selected ids.
def intention_filter(intentions: list[dict], ids: list[str] | None) -> list[dict]:
    """
    Filter the intention list to only include entries with selected ids.
    Args:
        intentions: Original intentions in format [{"intention_id":"001" ...}...]
        ids: List of selected ids (e.g., ["001", "003"])

    Returns:
        Filtered list containing only the specified intention dicts.
        If there is no ids input, the filtered list is empty.

    Raises:
        ValueError: If any id is not found in the list.
    """
    if not ids:
        return []

    # Build lookup dict: O(n)
    intention_map = {i["intention_id"]: i for i in intentions}

    # Find missing IDs: O(k)
    missing = [i for i in ids if i not in intention_map]
    if missing:
        e_m = f"以下意图id不存在：{missing}"
        logger_chatflow.error(e_m)
        raise ValueError(e_m)

    # Build filtered list in the same order as input IDs
    filtered = [intention_map[i] for i in ids]
    return filtered

# Update the target node id when setting up edges
def update_target(target: str, lookup: dict[str, str]) -> str:
    if target in list(lookup.keys()):
        target = lookup[target] # Convert target node id when switching main flow
    return f"{target}_reply"