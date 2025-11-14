from agent_builders.chatflow_builder import build_chatflow
from config.config_setup import ChatFlowConfig
from config.paths import AGENT_DATA_PATH, INTENTION_PATH, KNOWLEDGE_PATH, VECTOR_DB_PATH, CHATFLOW_DESIGN_PATH, \
    DIALOG_PATH, VECTOR_BD_COLLECTION_PATH
from functionals.log_utils import logger_chatflow
from functionals.state import ChatState

# Conversation config for one round of chat
conv_config = {"configurable": {"thread_id": "test_conv_1"}}

# The function to run the chatflow
def main():
    chatflow_config = ChatFlowConfig.from_paths(
        AGENT_DATA_PATH,
        KNOWLEDGE_PATH,
        CHATFLOW_DESIGN_PATH,
        INTENTION_PATH,
        DIALOG_PATH,
        VECTOR_BD_COLLECTION_PATH,
        VECTOR_DB_PATH
    )
    chatflow = build_chatflow(chatflow_config)

    print("=== 智能客服已上线 ===\n")

    # Initial state
    state = ChatState(
        messages=[],
        dialog_state=[],
        metadata=[{
            "role":"",
            "content":"",
            "intention_tag":"",
            "branch_count":{},
            "dialog_id":"",
            "logic":{
                "user_logic_title":{},
                "assistant_logic_title":"",
                "detail":{}
            }
        }]
    )

    # Step 1: Trigger welcome message
    state = chatflow.invoke(state, config=conv_config)

    # Print initial assistant message
    last_msg = state["messages"][-1]
    if last_msg["role"] == "assistant":
        logger_chatflow.info("智能客服：%s", {last_msg['content']})
        print(f"智能客服：{last_msg['content']}")
        previous_metadata = state["metadata"][-1]
        print("回复数据：\n" + "\n".join(f"{k}: {v}" for k, v in previous_metadata.items()))

    print()

    # Step 2: Main conversation loop
    while True:
        # Get user input
        user_input = input("用户：").strip()
        logger_chatflow.info("用户：%s", {user_input})
        if user_input.lower() == "挂电话":
            log_info = "用户已挂断电话"
            logger_chatflow.info("系统消息：%s", log_info)
            break

        # Inject user message
        state["messages"].append({"role": "user", "content": user_input})

        # Resume workflow
        try:
            state = chatflow.invoke(state, config=conv_config)
        except Exception as e:
            logger_chatflow.error("系统错误：%s", {e})
            break

        # Print and log new assistant messages
        for msg in state["messages"][-1:]:
            if msg["role"] == "assistant":
                logger_chatflow.info("智能客服：%s", {msg['content']})
                print(f"智能客服： {msg['content']}")
                metadata = state["metadata"][-1]
                print("回复数据：\n" + "\n".join(f"{k}: {v}" for k, v in metadata.items()))
        print()

    return state

if __name__ == "__main__":
    state = main()

    # Export state
    # def convert(obj):
    #     if isinstance(obj, set):
    #         return list(obj)
    #     return str(obj)  # fallback for other non-serializable types
    # with open("chat_state.json", "w", encoding="utf-8") as f:
    #     json.dump(state, f, ensure_ascii=False, indent=2, default=convert)

    # Print final messages
    print("=== 智能客服已下线 ===")
    print("聊天记录：")
    for chat in state["messages"]:
        print(f"{chat['role']}: {chat['content']}")
    print("元数据：")
    for metadata in state["metadata"]:
        print(metadata)