from functionals.log_utils import logger_chatflow
from functionals.state import ChatState

# Function for the hang_up node in every project
def hang_up(state: ChatState) -> dict:
    log_info = "智能客服已挂断电话"
    logger_chatflow.info("系统消息：%s", log_info)
    return {
        "messages": state["messages"],
        "dialog_state": None,
        "logs": state["logs"],
        "metadata": state["metadata"]
    }