from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage, HumanMessage, AIMessage

basic_model = init_chat_model(model="ollama:nemotron3:33b", temperature=0.15)
advanced_model = init_chat_model(model="ollama:qwen3:14b", temperature=0.15)


@wrap_model_call
def dynamic_model_selector_middleware(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])
    selected_model = basic_model if message_count < 3 else advanced_model
    request = request.override(model=selected_model)

    return handler(request)


if __name__ == "__main__":
    agent = create_agent(
        model=basic_model, middleware=[dynamic_model_selector_middleware]
    )
    response = agent.invoke(
        {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is the capital of France?"),
                # HumanMessage(content="What is the capital of Germany?"),
                # HumanMessage(content="What is the capital of Italy?"),
                # HumanMessage(content="What is the capital of Spain?"),
            ]
        }
    )
    print(response)
    print("---------------------------------")
    print(response["messages"][-1].content)
    print("---------------------------------")
    print(response["messages"][-1].response_metadata["model"])
