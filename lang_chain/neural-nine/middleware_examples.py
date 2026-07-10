from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt


@dataclass
class Context:
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = "You are a helpful and very concise assistant."

    match user_role:
        case "export":
            return f"{base_prompt} Provide detailed technical responses."
        case "beginner":
            return f"{base_prompt} Keep your explanations simple and easy to understand, avoiding technical jargon."
        case "child":
            return f"{base_prompt} Explain everything as if you were literally talking to a five-year-old child. Use simple language and concepts."
        case _:
            return base_prompt


if __name__ == "__main__":
    agent = create_agent(
        model="ollama:qwen3:14b", middleware=[user_role_prompt], context_schema=Context
    )
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain PCA.",
                }
            ]
        },
        context=Context(user_role="beginner"),
    )
    print(response)
    print("------------------------------")
    print(response["messages"][-1].content)
