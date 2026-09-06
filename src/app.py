import chainlit as cl
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import chainlit.data as cl_data


load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")

cl_data._data_layer = SQLAlchemyDataLayer(conninfo=DATABASE_URL)

API_KEY = os.environ.get("API_KEY")
client = AsyncOpenAI(
    api_key=API_KEY.strip(),
    base_url="https://api.metisai.ir/openai/v1",
)

'''
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
'''

SYSTEM_PROMPT = """
شما یک دستیار هوشمند و مودب برای راهنمایی دانشجویان دانشگاه صنعتی اصفهان هستید.
وظیفه شما پاسخگویی به سوالات دانشجویان درباره لیست دروس، اساتید، واحدها،مکان ها و ظرفیت‌هاست.
قوانین:
- فقط بر اساس اطلاعاتی (Context) که به شما داده می‌شود پاسخ دهید.
- اگر جواب سوال در اطلاعات داده شده نبود، بگویید "متاسفانه اطلاعاتی در این باره ندارم."
- پاسخ‌ها را خوانا، کوتاه و در صورت نیاز به صورت بولت‌پوینت بنویسید.
"""


@cl.password_auth_callback
async def auth(username, password):
    # To Do: real authentication
    return cl.User(identifier=username)


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
    new_temp = settings["Temperature"]
    cl.user_session.set("temperature", new_temp)
    print(f"Temperature updated to: {new_temp}")


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for step in thread["steps"]:
        if step["type"] == "user_message":
            history.append({"role": "user", "content": step["output"]})
        elif step["type"] == "assistant_message":
            history.append({"role": "assistant", "content": step["output"]})
    cl.user_session.set("message_history", history)
    cl.user_session.set("temperature", 1.0)
    print(f"Chat resumed! Loaded {len(history)} messages from database.")


@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    history.append({"role": "user", "content": message.content})
    temp = cl.user_session.get("temperature")

    msg = cl.Message(content="")
    await msg.send()

    stream = await client.chat.completions.create(
        model="gpt-4o-mini",  # "gpt-4o-mini"
        messages=history,
        temperature=temp,
        stream=True,
    )

    full_response = ""

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            await msg.stream_token(token)

    history.append({"role": "assistant", "content": full_response})

    await msg.update()
