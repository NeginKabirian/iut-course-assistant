
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct , Filter, FieldCondition, MatchValue

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

load_dotenv()
API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL", "https://api.metisai.ir/openai/v1")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")

print("Connecting to OpenAI and Qdrant...")
client = OpenAI(api_key=API_KEY.strip(), base_url=BASE_URL.strip())
qdrant = QdrantClient(url=QDRANT_URL)
COLLECTION_NAME = "iut_courses"


if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(COLLECTION_NAME,points_selector=Filter(must=[FieldCondition(key="type", match=MatchValue(value="course"))]),)
    print(f"Old collection '{COLLECTION_NAME}' deleted.")

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
print(f"Collection '{COLLECTION_NAME}' created successfully.\n")


print("Reading and processing Excel file...")
df = pd.read_excel("courses.xlsx", skiprows=2, header=None)

df.columns = [
    "code",
    "name",
    "total_units",
    "prac_units",
    "capacity",
    "registered",
    "waitlist",
    "gender",
    "prof_name",
    "prof_type",
    "prof_role",
    "time_loc",
    "description",
]

header_repeat_idx = df.index[df["code"] == "شماره و گروه درس"].tolist()
rows_to_drop = set()
for i in header_repeat_idx:
    rows_to_drop.update(range(max(i - 2, 0), i + 2))

df = df.drop(index=[i for i in rows_to_drop if i in df.index]).reset_index(drop=True)
print(f"Removed {len(rows_to_drop)} repeated page-header rows ({len(header_repeat_idx)} blocks).\n")

group_id = df["code"].notna().cumsum()

cols_to_fill = [
    "code",
    "name",
    "total_units",
    "prac_units",
    "capacity",
    "registered",
    "waitlist",
    "gender",
    "prof_name",  
    "prof_type", 
    "prof_role", 
    "description",
]
df[cols_to_fill] = df.groupby(group_id)[cols_to_fill].ffill()
df = df.fillna("")

grouped_courses = df.groupby("code")
documents = []

for code, group in grouped_courses:
    first_row = group.iloc[0]

    base_text = (
        f"درس '{first_row['name']}' در دانشگاه صنعتی اصفهان (IUT) با شماره گروه {code}، "
        f"تعداد واحد کل: {str(first_row['total_units']).replace('.0', '')}، "
        f"ظرفیت کل: {str(first_row['capacity']).replace('.0', '')} نفر، "
        f"تعداد ثبت‌نام شده: {str(first_row['registered']).replace('.0', '')} نفر، "
        f"لیست انتظار: {str(first_row['waitlist']).replace('.0', '')} نفر، "
        f"جنسیت مجاز: {first_row['gender']}. "
    )

    prof_dict = {}
    for index, row in group.iterrows():
        prof = str(row["prof_name"]).strip()
        role = str(row["prof_role"]).strip()
        time_l = str(row["time_loc"]).strip().replace('\n', ' ').replace('\r', '')

        if prof:
            prof_key = f"{prof} (مسئولیت: {role})"
            if prof_key not in prof_dict:
                prof_dict[prof_key] = []
            
            if time_l:
                prof_dict[prof_key].append(time_l)

  
    prof_details = []
    for p_key, times in prof_dict.items():
        if times:
            
            times_str = " و ".join(times) 
            prof_details.append(f"استاد/دستیار: {p_key} | زمان و مکان: {times_str}")
        else:
            prof_details.append(f"استاد/دستیار: {p_key}")

    if prof_details:
        base_text += "جزئیات اساتید و زمان برگزاری: [" + " ؛ ".join(prof_details) + "]. "

    if str(first_row["description"]).strip():
        base_text += f"توضیحات تکمیلی: {first_row['description']}"

    documents.append({"text": base_text, "course_name": first_row["name"]})

print(f"Extracted {len(documents)} rich courses.\n")
print("Starting embedding process (this might take a minute)...")
BATCH_SIZE = 50
total_indexed = 0

for i in range(0, len(documents), BATCH_SIZE):
    batch_docs = documents[i : i + BATCH_SIZE]
    texts_to_embed = [doc["text"] for doc in batch_docs]
    response = client.embeddings.create(
        input=texts_to_embed, model="text-embedding-3-small"
    )

    points = []
    for j, res in enumerate(response.data):
        point = PointStruct(
            id=total_indexed + j + 1,
            vector=res.embedding,
            payload={
                "text": batch_docs[j]["text"],
                "course_name": batch_docs[j]["course_name"],
                "type": "course",
            },
        )
        points.append(point)

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    total_indexed += len(points)
    print(f"Indexed {total_indexed}/{len(documents)} courses...")

print("\nAll rich data successfully embedded and stored in Qdrant!")
