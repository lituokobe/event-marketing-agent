from typing import TypedDict, Annotated

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
    metadata: the list of metadata. metadata is only added when there is a reply (usually in a reply node)
    metadata structure:{
                        "dialog_id":str,
                        "content":str,
                        "main_flow_id":str,
                        "main_flow_name":str,
                        "node_id":str,
                        "node_name":str,
                        "intention_approach":str,
                        "i_a_details":str,
                        "end_call":bool
                        }
    """
    messages: list[dict]
    dialog_state: Annotated[
        list[str|None],
        update_dialog_stack
    ]
    metadata: list[dict]