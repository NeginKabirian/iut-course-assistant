from core.clients import async_openai_client
from rag.prompts import get_router_prompt

async def analyze_user_intent(user_query: str) -> str:
   
    router_prompt = get_router_prompt(user_query)
    
    try:
        response = await async_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=router_prompt,
            temperature=0.0, 
            max_tokens=10
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"Router Error: {e}")
        return "SEARCH"

    