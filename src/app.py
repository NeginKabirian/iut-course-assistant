import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import chainlit.data as cl_data

from core.config import DATABASE_URL, DEFAULT_TEMPERATURE
from rag.prompts import SYSTEM_PROMPT
from rag.generator import process_user_query

cl_data._data_layer = SQLAlchemyDataLayer(conninfo=DATABASE_URL)

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

@cl.on_message
async def main(message: cl.Message):
    try:
        await process_user_query(message.content)
    except Exception as e:
        print(f"Error processing user query: {e}")
        await cl.Message(
            content="متاسفانه در پردازش سوال شما مشکلی پیش آمد. لطفاً دوباره تلاش کنید."
        ).send()

@cl.action_callback("suggested_question")
async def on_suggestion_click(action: cl.Action):
    await cl.Message(content=action.payload["value"], author="User").send()
    await action.remove()
    try:
        await process_user_query(action.payload["value"], is_suggestion=True)
    except Exception as e:
        print(f"Error processing suggested question: {e}")
        await cl.Message(
            content="متاسفانه در پردازش سوال شما مشکلی پیش آمد. لطفاً دوباره تلاش کنید."
        ).send()