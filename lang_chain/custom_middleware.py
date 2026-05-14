from dataclasses import dataclass
import time

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import SystemMessage, HumanMessage, AIMessage


class HookMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.agent_start_time = 0.0
        self.model_start_time = 0.0

    def before_agent(self, state: AgentState, runtime):
        self.agent_start_time = time.time()
        print("before_agent triggered")

    def before_model(self, state: AgentState, runtime):
        self.model_start_time = time.time()
        print("before_model triggered")

    def after_model(self, state: AgentState, runtime):
        print(
            "after_model triggered, Execution time: {:.2f} seconds".format(
                time.time() - self.model_start_time
            )
        )

    def after_agent(self, state: AgentState, runtime):
        print(
            "after_agent triggered, Execution time: {:.2f} seconds".format(
                time.time() - self.agent_start_time
            )
        )


if __name__ == "__main__":
    model = init_chat_model(model="ollama:nemotron3:33b", temperature=0.15)
    agent = create_agent(model=model, middleware=[HookMiddleware()])

    response = agent.invoke(
        {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is the capital of France?"),
            ]
        }
    )
    print("---------------------------------")
    print(response)
    print("---------------------------------")
    print(response["messages"][-1].content)
