from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent

emmeddings = OllamaEmbeddings(model="qwen3-embedding:8b")


def ollama_embeddings():
    texts = [
        "Apple makes verry good computers.",
        "I believe Apple is innovative!",
        "I love Apple.",
        "I am a fan of MacBooks.",
        "I enjoy oranges.",
        "I like Lenovo ThinkPads.",
        "I think pears taste very good.",
    ]

    vector_store = FAISS.from_texts(texts, embedding=emmeddings)
    print(vector_store.similarity_search("Apples are my favorite fruit.", k=7))
    print(vector_store.similarity_search("Linux is a great operating system.", k=7))


def ollama_embeddings_with_faiss():
    texts = [
        "I love apples.",
        "I enjoy oranges.",
        "I think pears taste very good.",
        "I hate bananas.",
        "I dislike raspberries.",
        "I dispise mangoes.",
        "I love Linux.",
        "I hate Windows.",
    ]

    vector_store = FAISS.from_texts(texts, embedding=emmeddings)

    print(vector_store.similarity_search("what fruits does the person like?", k=3))
    print(vector_store.similarity_search("What fruits does the person dislike?", k=3))


def ollama_embeddings_with_retriever():
    texts = [
        "I love apples.",
        "I enjoy oranges.",
        "I think pears taste very good.",
        "I hate bananas.",
        "I dislike raspberries.",
        "I dispise mangoes.",
        "I love Linux.",
        "I hate Windows.",
    ]

    vector_store = FAISS.from_texts(texts, embedding=emmeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retriever_tool = create_retriever_tool(
        retriever,
        name="kb_search",
        description="Search the small product / fruit knowledge base for information",
    )

    agent = create_agent(
        model="ollama:qwen3:14b",
        tools=[retriever_tool],
        system_prompt="You are a helpful assistant. For questions about fruits or operating systems, you should use the kb_search tool to find relevant information from the knowledge base.",
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What three fruits does the person like and dislike?",
                }
            ]
        }
    )
    print(result)
    print("------------------------------")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    # ollama_embeddings()
    # ollama_embeddings_with_faiss()
    ollama_embeddings_with_retriever()
