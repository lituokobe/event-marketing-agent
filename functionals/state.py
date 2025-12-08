from typing import TypedDict, Annotated
from langgraph.graph import add_messages

# Reducer function to edit ChatState
def update_dialog_stack(left: list[str], right:str|None)->list[str]:
    """
    Update the dialog state stack
    :param left: current state stack
    :param right: new state or action to add to the stack. If none, no action;
                  if 'pop', pop up the top (last one) of the stack; otherwise add it to the stack.
    :return: updates stack
    """
    if right is None:
        return left
    if right == 'pop':
        return left[:-1] #remove the last one of the stack
    if isinstance(right, list):
        return left + right
    if isinstance(right, str):
        return left + [right]
    return left  # fallback

class ChatState(TypedDict):
    """
    state class:
    messages: a list of chat history from both the user and the agent
    dialog_state: a list of node names that indicate the direction of the chatflow
    logs: a list of logs to document the chatflow information
    metadata: the list of metadata. metadata is only added when there is a reply (usually in a reply node)
    logs=[{
        "role": "",
        “content": "",
        "main_flow_id": "",
        "main_flow_name": "",
        "node_id": "",
        "node_name": "",
        "match_to": "",
        "branch_id": "",
        "branch_name": "",
        "branch_name_count": {},
        "intention_id": "",
        "intention_name": "",
        "infer_tool": "",
        "llm_input_summary": "",
        "matching_content": "",
        "matching_score": 0.0,
        "knowledge_type": "",
        "time_cost": 0.0
    }],
    metadata=[{
        "role":"",
        "content":"",
        "dialog_id":"",
        ”end_call": bool
        "logic":{
            "user_logic_title":{},
            "assistant_logic_title":"",
            "complete_process":[],
            "detail":[]
        }
    }]
    """
    messages: Annotated[list[dict], add_messages]
    # Very important to have add_messages. This means the input of invoke can be just a message, but the graph will merge it to the existing state
    dialog_state: Annotated[
        list[str|None],
        update_dialog_stack
    ]
    logs: list[dict|None]
    metadata: list[dict|None]
