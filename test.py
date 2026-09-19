import pytest
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"


load_dotenv()
API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")

client = AsyncOpenAI(api_key=API_KEY.strip(), base_url=BASE_URL.strip())
qdrant = AsyncQdrantClient(url="http://127.0.0.1:6333")
COLLECTION_NAME = "iut_courses"


@pytest.mark.asyncio
async def test_rag_course_capacity():
    user_query = "ظرفیت کلاس امنیت شبکه استاد فانیان چقدر است؟"

    embeded_res = await client.embeddings.create(
        input=user_query, model="text-embedding-3-small"
    )
    query_vector = embeded_res.data[0].embedding

    search_res = await qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=3
    )
    context_text = "\n".join([record.payload["text"] for record in search_res.points])

    assert "فانيان" in context_text
    assert "50" in context_text or "۵۰"

    prompt = f"اطلاعات: {context_text}\n\nسوال:{user_query}"

    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0
    )

    answer = response.choices[0].message.content
    assert "50" in answer or "۵۰" in answer


@pytest.mark.asyncio
async def test_prerequisites_rules():

    user_query = "برای برداشتن درس ساختمان داده، چه درسی را باید قبلا پاس کرده باشم؟"
    embed_res = await client.embeddings.create(
        input=user_query, model="text-embedding-3-small"
    )
    query_vector = embed_res.data[0].embedding
    search_res = await qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=3
    )
    context_text = "\n".join([record.payload["text"] for record in search_res.points])
    prompt = f"اطلاعات: {context_text}\n\nسوال: {user_query}"
    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    answer = response.choices[0].message.content
    assert "برنامه‌سازی پیشرفته" in answer or "برنامه سازی پیشرفته" in answer
    assert "ساختمان‌های گسسته" in answer or "ساختمان های گسسته" in answer


@pytest.mark.asyncio
async def test_rag_course_instructor():
    user_query = "استاد درس نرم افزار 1 کیست؟"

    embeded_res = await client.embeddings.create(
        input=user_query, model="text-embedding-3-small"
    )
    query_vector = embeded_res.data[0].embedding

    search_res = await qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=3
    )
    context_text = "\n".join([record.payload["text"] for record in search_res.points])

    assert "نصر اصفهاني مريم" in context_text
    assert "بقولي زاده شيرين" in context_text

    prompt = f"اطلاعات: {context_text}\n\nسوال:{user_query}"

    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0
    )

    answer = response.choices[0].message.content
    assert  "نصر اصفهاني مريم" in answer or "بقولي زاده شيرين" in answer
