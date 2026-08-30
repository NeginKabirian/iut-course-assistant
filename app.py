
import chainlit as cl

@cl.on_message
async def main(message: cl.Message):
    content = message.content
    reply = f"Echo Bot: {content}"
    await cl.Message(content=reply).send()


