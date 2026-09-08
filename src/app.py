import chainlit as cl
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import chainlit.data as cl_data
from qdrant_client import AsyncQdrantClient

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
BASE_URL = os.environ.get("BASE_URL")

cl_data._data_layer = SQLAlchemyDataLayer(conninfo=DATABASE_URL)

API_KEY = os.environ.get("API_KEY")
client = AsyncOpenAI(
    api_key=API_KEY.strip(),
    base_url=BASE_URL.strip(),
)
qdrant = AsyncQdrantClient(url="http://127.0.0.1:6333")
COLLECTION_NAME = "iut_courses"


SYSTEM_PROMPT = """
شما یک دستیار هوشمند و مودب برای راهنمایی دانشجویان دانشگاه صنعتی اصفهان هستید.
وظیفه شما پاسخگویی به سوالات دانشجویان درباره لیست دروس، اساتید، واحدها،مکانها و ظرفیت‌هاست.
قوانین:
- فقط بر اساس اطلاعاتی (Context) که به شما داده می‌شود پاسخ دهید.
- .اگر جواب سوال در اطلاعات داده شده نبود، بگویید "متاسفانه اطلاعاتی در این باره ندارم"
- .پاسخ‌ها را خوانا، کوتاه و در صورت نیاز به صورت بولت‌پوینت بنویسید
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
                initial=0.1,
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
    temp = cl.user_session.get("temperature")

    user_query = message.content
    search_query = user_query

    if len(history) > 1:
        recent_history = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in history[-5:]
                if msg["role"] != "system"
            ]
        )

        rewrite_prompt = [
            {
                "role": "system",
                "content": "تو یک دستیار هوشمند هستی. وظیفه تو این است که با توجه به تاریخچه مکالمه، سوالِ جدیدِ کاربر را به یک سوال کامل، شفاف و مستقل تبدیل کنی که بدون نیاز به تاریخچه قابل جستجو باشد. فقط خودِ سوال را بنویس و هیچ کلمه اضافه‌ای نگو.",
            },
            {
                "role": "user",
                "content": f"تاریخچه:\n{recent_history}\n\nسوال جدید کاربر: {user_query}\n\nسوال مستقل و کامل:",
            },
        ]
        rewrite_response = await client.chat.completions.create(
            model="gpt-4o-mini", messages=rewrite_prompt, temperature=0.0, stream=False
        )
        search_query = rewrite_response.choices[0].message.content.strip()
        print(f"Original: {user_query} | Rewritten: {search_query}")
    embed_user_query = await client.embeddings.create(
        input=search_query, model="text-embedding-3-small"
    )
    query_vector = embed_user_query.data[0].embedding
    search_res = await qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=8
    )

    context_text = "\n".join([record.payload["text"] for record in search_res.points])

    rag_prompt = f""" 
    اطلاعات زیر از دیتابیس دانشگاه استخراج شده است:
    {context_text}
    
    با توجه به اطلاعات بالا به این سوال پاسخ بده:
    {search_query}
    """
    history.append({"role": "user", "content": rag_prompt})
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

    history[-1]["content"] = user_query
    history.append({"role": "assistant", "content": full_response})

    await msg.update()
