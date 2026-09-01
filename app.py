import chainlit as cl
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("API_KEY")
client = AsyncOpenAI(
    api_key=API_KEY.strip(),
    base_url="https://api.metisai.ir/openai/v1",
)

SYSTEM_PROMPT = """
You are an elite Senior Technical Recruiter and Career Coach specializing in the Tech & Software Engineering industry.
Your main goals are:
1. Conduct mock technical and behavioral (STAR method) interviews.
2. Provide ruthless but constructive resume reviews.
3. Guide users on career roadmaps and modern tech stacks (e.g., Python, FastAPI, AI, DevOps).

Rules you MUST follow:
- Maintain a professional, encouraging, and highly technical tone.
- If the user asks non-career or non-tech related questions (like cooking, sports, etc.), politely decline and steer the conversation back to career coaching.
- When reviewing or answering, provide specific, actionable feedback rather than generic advice.
- Keep your answers concise and well-formatted using markdown (bullet points, bold text).
"""


@cl.on_chat_start
async def start():
    settings = cl.ChatSettings(
        [
            cl.input_widget.Slider(
                id="Temperature",
                label="Temperature",
                initial=1.0,
                min=0.0,
                max=2.0,
                step=0.1,
            )
        ]
    )
    await settings.send()
    cl.user_session.set("message_history", [{"role": "system", "content": SYSTEM_PROMPT}])
    cl.user_session.set("temperature", 1.0)


@cl.on_settings_update
async def setup_agent(settings):
    pass


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    history.append({"role": "user", "content": message.content})
    temp = cl.user_session.get("temperature")

    msg = cl.Message(content="Thinking...")
    await msg.send()

    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=history,
        temperature=temp,
        stream=False,
    )

    answer = response.choices[0].message.content

    history.append({"role": "assistant", "content": str(answer)})
    msg.content = str(answer)

    await msg.update()
