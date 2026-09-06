import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")

client = OpenAI(api_key=API_KEY.strip(), base_url=BASE_URL.strip())

qdrant = QdrantClient(url="http://127.0.0.1:6333")

COLLECTION_NAME = "iut_courses"


if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")

sample_documents = [
    "درس مبانی مکاترونیک با شماره گروه ۰۱_۱۷۱۶۴۳۵، به ارزش ۳ واحد، با ظرفیت ۲۵ نفر برای جنسیت مختلط توسط استاد جلالی بیدگلی حامد ارائه می‌شود.",
    "درس سیستم‌های کنترل خطی با شماره گروه ۰۱_۱۷۱۶۳۱۰، به ارزش ۳ واحد، با ظرفیت ۵۰ نفر توسط استاد شیخ الاسلام فرید ارائه می‌شود.",
    "درس آزمایشگاه سیستم عامل با شماره گروه ۰۱_۱۷۳۴۳۰۴ توسط استاد زالی زینب ارائه می‌شود و ظرفیت آن ۱۶ نفر است.",
]

points = []
for i, doc in enumerate(sample_documents):
    response = client.embeddings.create(input=doc, model="text-embedding-3-small")
    embedding_vector = response.data[0].embedding

    point = PointStruct(id=i + 1, vector=embedding_vector, payload={"text": doc})
    points.append(point)

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"Successfully indexed {len(points)} documents into Qdrant!")
