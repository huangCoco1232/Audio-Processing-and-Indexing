# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import pipe

app = FastAPI()


# 请求结构
class ChatRequest(BaseModel):
    messages: list


# 自定义 system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a conversational AI designed for voice output. "
    "You must respond using only plain natural sentences without any formatting. "
    "Do not output \\n or newline characters under any circumstances. "
    "Do not use markdown. Do not use bullet points. Do not use numbered lists. "
    "Do not output any symbols such as *, -, #, >, :, ;, or backticks. "
    "Do not start sentences with numbers like 1, 2, 3. "
    "using natural language like 'for example' or 'another option is', not lists. "
    "Always output response in plain text only."
    "Keep your responses concise and no longer than 2–3 sentences. This is important. "
    "Do not produce long paragraphs. "
)

full_messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]


@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.messages[0]

    # 附加 system prompt
    full_messages.append({"role": "user", "content": user_message})

    print(full_messages)

    generation_args = {
        "max_new_tokens": 100,
        "return_full_text": False,
        "temperature": 0.2,
    }

    output = pipe(full_messages, **generation_args)
    reply = output[0]["generated_text"]

    full_messages.append({"role": "system", "content": reply})

    return {"reply": reply}