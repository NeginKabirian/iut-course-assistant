import json
import chainlit as cl
from core.clients import async_openai_client, async_qdrant_client
from core.config import COLLECTION_NAME, MAX_HISTORY
from rag.prompts import get_rewrite_prompt, get_suggestion_prompt, get_rag_prompt
from rag.intent_router import analyze_user_intent

async def generate_suggestions(context_text: str, answer: str) -> list:
    prompt = get_suggestion_prompt(context_text, answer)
    try:
        res = await async_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        return [q.strip() for q in data.get("questions", []) if q and q.strip()][:3]
    except Exception as e:
        print(f"Suggestion Error: {e}")
        return []


async def process_user_query(user_query: str, is_suggestion: bool = False):

    if not is_suggestion:
        intent = await analyze_user_intent(user_query)
        
        if "GREETING" in intent:
            await cl.Message(
                content="سلام! 👋 من دستیار هوشمند دانشگاه صنعتی اصفهان هستم. چطور می‌تونم درباره دروس، زمان برگزاری کلاس‌ها و اساتید راهنماییتون کنم؟"
            ).send()
            return  
            
        elif "OUT" in intent:
            await cl.Message(
                content="من فقط برای پاسخگویی به سوالات آموزشی دانشگاه صنعتی اصفهان (دروس، اساتید و کلاس‌ها) طراحی شده‌ام. لطفاً سوال مرتبط بپرسید."
            ).send()
            return  

    history = cl.user_session.get("message_history")
    temp = cl.user_session.get("temperature")
    search_query = user_query

    if len(history) > 1 and not is_suggestion:
        recent_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history[-5:] if msg["role"] != "system"]
        )
        rewrite_prompt = get_rewrite_prompt(recent_history, user_query)
       
        rewrite_response = await async_openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=rewrite_prompt, temperature=0.0, stream=False
        )
        search_query = rewrite_response.choices[0].message.content.strip()

    search_query = search_query.replace("ي", "ی").replace("ك", "ک")

    embed_user_query = await async_openai_client.embeddings.create(
        input=search_query, model="text-embedding-3-small"
    )
    query_vector = embed_user_query.data[0].embedding
    
    search_res = await async_qdrant_client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=15
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
    
    rag_prompt = get_rag_prompt(context_text, search_query)
    history.append({"role": "user", "content": rag_prompt})

    msg = cl.Message(content="")
    await msg.send()

    stream = await async_openai_client.chat.completions.create(
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
    for q in await generate_suggestions(context_text, full_response):
        actions.append(cl.Action(name="suggested_question", payload={"value": q}, label=f"💡{q}"))
    msg.actions = actions
    await msg.update()