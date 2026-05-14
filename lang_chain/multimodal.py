from base64 import b64encode
import httpx

from langchain.chat_models import init_chat_model


def describe_img_url(model, img_url):
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of the image."},
            {"type": "image", "url": img_url},
        ],
    }
    response = model.invoke([message])
    return response.content


def describe_img_base64(model, img_path):
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of the image."},
            {
                "type": "image",
                "base64": b64encode(open(img_path, "rb").read()).decode("utf-8"),
                "mime_type": "image/jpeg",
            },
        ],
    }
    response = model.invoke([message])
    return response.content


if __name__ == "__main__":
    model = init_chat_model("ollama:nemotron3:33b", temperature=0.15)
    # response = describe_img_url(
    #     model,
    #     "https://cdn.britannica.com/74/252374-050-AD45E98E/dog-breed-height-comparison.jpg",
    # )
    # print(response)

    response = describe_img_base64(
        model,
        "lang_chain/ext/dog-breed-height-comparison.jpg",
    )
    print(response)
