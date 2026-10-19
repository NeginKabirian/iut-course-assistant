from langfuse.openai import AsyncOpenAI
from openai import OpenAI
from qdrant_client import AsyncQdrantClient, QdrantClient
from core.config import API_KEY, BASE_URL, QDRANT_URL


async_openai_client = AsyncOpenAI(
    api_key=API_KEY.strip(),
    base_url=BASE_URL.strip(),
)

async_qdrant_client = AsyncQdrantClient(url=QDRANT_URL)


sync_openai_client = OpenAI(
    api_key=API_KEY.strip(),
    base_url=BASE_URL.strip(),
)

sync_qdrant_client = QdrantClient(url=QDRANT_URL)
