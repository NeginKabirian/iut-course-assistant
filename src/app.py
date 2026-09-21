import chainlit as cl
from langfuse.openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import chainlit.data as cl_data
from qdrant_client import AsyncQdrantClient
import json

load_dotenv()

for k in ("DATABASE_URL", "BASE_URL", "API_KEY"):
    if not os.environ.get(k):
        raise RuntimeError(f"Missing environment variable: {k}")

DATABASE_URL = os.environ.get("DATABASE_URL")
BASE_URL = os.environ.get("BASE_URL")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_TEMPERATURE = 0.1
MAX_HISTORY = 12

cl_data._data_layer = SQLAlchemyDataLayer(conninfo=DATABASE_URL)
API_KEY = os.environ.get("API_KEY")
client = AsyncOpenAI(
    api_key=API_KEY.strip(),
    base_url=BASE_URL.strip(),
)
qdrant = AsyncQdrantClient(url=QDRANT_URL)
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


async def start_settings():
    settings = cl.ChatSettings([
        cl.input_widget.Slider(id="Temperature", label="Temperature",
                               initial=DEFAULT_TEMPERATURE, min=0.0, max=2.0, step=0.1)
    ])
    await settings.send()
@cl.on_chat_start
async def start():
    await start_settings()
    cl.user_session.set("message_history", [{"role": "system", "content": SYSTEM_PROMPT}])
    cl.user_session.set("temperature", DEFAULT_TEMPERATURE)


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
    cl.user_session.set("temperature", DEFAULT_TEMPERATURE)
    print(f"Chat resumed! Loaded {len(history)} messages from database.")

async def analyze_user_intent(user_query: str) -> str:
    router_prompt = [
        {
            "role": "system",
            "content": 
            """تو یک سیستم مسیریاب (Router) هوشمند هستی. 
            :پیام کاربر را بخوان و فقط و فقط یکی از ۳ کلمه زیر را برگردان (هیچ کلمه اضافه‌ای نگو)

            - GREETING : اگر پیام سلام، احوال‌پرسی، تشکر است، یا کاربر می‌پرسد "چه کاری می‌توانی انجام دهی؟" یا "درباره چه چیزی اطلاع داری؟"
            - OUT : .اگر پیام توهین‌آمیز است یا موضوع آن هیچ ربطی به دانشگاه، انتخاب واحد، اساتید و دروس ندارد (مثل آشپزی، سیاست، جوک)
            - SEARCH : .اگر پیام درباره دروس، اساتید، زمان برگزاری، ظرفیت، پیش‌نیازها و مسائل دانشگاهی است

            .فقط یک کلمه به انگلیسی جواب بده
            """
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=router_prompt,
            temperature=0.0, 
            max_tokens=5
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"Router Error: {e}")
        return "SEARCH"
async def generate_suggestions(context_text:str,answer: str) -> list:
    prompt = [
        {
            "role": "system",
            "content": """تو فقط سوال پیشنهادی تولید می‌کنی.
            قوانین سخت‌گیرانه:
            - فقط سوالی بنویس که پاسخ کامل و صریح آن در «متن زمینه» موجود باشد.
            - هر سوال باید مستقل و کامل باشد (اسم درس یا استاد را داخل خودش بیاور، از «آن درس» استفاده نکن).
            - سوالی که در «پاسخ داده‌شده» جوابش آمده را تکرار نکن.
            - اگر هیچ سوال مناسبی وجود ندارد، آرایه خالی برگردان.
            خروجی فقط JSON به این شکل: {"questions": ["...", "..."]} — حداکثر3 مورد.""",
                    },
        {
            "role": "user",
            "content": f"متن زمینه:\n{context_text}\n\nپاسخ داده‌شده:\n{answer}",
        },
    ]
    try:
        res = await client.chat.completions.create(model="gpt-4o-mini",messages=prompt,temperature=0.0,response_format={"type":"json_object"},)
        data = json.loads(res.choices[0].message.content)
        return [q.strip() for q in data.get("questions",[] )if q and q.strip()][:3]
    except Exception as e:
        print(f"Suggestion Error: {e}")
        return []
async def process_user_query(user_query: str,is_suggestion: bool = False ):
    if not is_suggestion:
        intent = await analyze_user_intent(user_query)
        
        if "GREETING" in intent:
            await cl.Message(
                content="سلام! 👋 من دستیار هوشمند دانشگاه صنعتی اصفهان هستم. چطور می‌تونم درباره دروس، زمان برگزاری کلاس‌ها و اساتید راهنماییتون کنم؟"
            ).send()
            return  
            
        elif "OUT" in intent:
            await cl.Message(
                content=".من فقط برای پاسخگویی به سوالات آموزشی دانشگاه صنعتی اصفهان (دروس، اساتید و کلاس‌ها) طراحی شده‌ام. لطفاً سوال مرتبط بپرسید"
            ).send()
            return  


    history = cl.user_session.get("message_history")
    temp = cl.user_session.get("temperature")
    search_query = user_query

    if len(history) > 1 and not is_suggestion:
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
                "content": "تو یک دستیار هوشمند هستی. وظیفه تو این است که با توجه به تاریخچه مکالمه، سوالِ جدید کاربر را به یک سوال کامل، شفاف و مستقل تبدیل کنی که بدون نیاز به تاریخچه قابل جستجو باشد. فقط خود سوال را بنویس و هیچ کلمه اضافه‌ای نگو",
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

    embed_user_query = await client.embeddings.create(
        input=search_query, model="text-embedding-3-small"
    )
    query_vector = embed_user_query.data[0].embedding
    search_res = await qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=8
    )

    fresh_context = "\n".join(
        [r.payload.get("text", "") for r in search_res.points if r.payload.get("text")]
    )

    context_text = fresh_context
    if is_suggestion:
        prev = cl.user_session.get("last_context") or ""
        if prev:
            seen = set(fresh_context.split("\n"))
            extra = [ln for ln in prev.split("\n") if ln and ln not in seen]
            if extra:
                context_text = fresh_context + "\n" + "\n".join(extra)

    cl.user_session.set("last_context", fresh_context)
    rag_prompt = f""" 
    اطلاعات زیر از دیتابیس دانشگاه استخراج شده است:
    {context_text}
    
    با توجه به اطلاعات بالا به این سوال پاسخ بده:
    {search_query}
        
    قوانین بسیار مهم: 
    ۱. اگر کاربر درباره اساتید، گروه‌های ارائه‌دهنده، ظرفیت یا زمان برگزاری یک درس پرسید، تو موظف هستی متن را کامل بگردی و اطلاعات **"تمامی گروه‌ها و تمام اساتیدی"** که آن درس را ارائه می‌دهند لیست کنی. هرگز به اولین موردی که پیدا کردی اکتفا نکن!
    """
    history.append({"role": "user", "content": rag_prompt})

    msg = cl.Message(content="")
    await msg.send()

    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
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
    if len(history) > MAX_HISTORY:
        del history[1:len(history) - (MAX_HISTORY - 1)]
    actions = []
    for q in await generate_suggestions(context_text,full_response):
        actions.append(cl.Action(name="suggested_question",payload={"value":q},label=f"💡{q}"))
    msg.actions = actions
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    try:
        await process_user_query(message.content)
    except Exception as e:
        print(f"Error processing user query: {e}")
        await cl.Message(
            content=".متاسفانه در پردازش سوال شما مشکلی پیش آمد.لطفاً دوباره تلاش کنید"
        ).send()

@cl.action_callback("suggested_question")
async def on_suggestion_click(action: cl.Action):
    await cl.Message(content=action.payload["value"], author="User", type="user_message").send()
    await action.remove()
    try:
        await process_user_query(action.payload["value"],is_suggestion=True)
    except Exception as e:
        print(f"Error processing suggested question: {e}")
        await cl.Message(
            content=".متاسفانه در پردازش سوال شما مشکلی پیش آمد.لطفاً دوباره تلاش کنید"
        ).send()
