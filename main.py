import json
import os
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Initialize FastAPI App
app = FastAPI(title="Electronyatk AI Store & Project Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Ensure static/images/<category> folders exist and mount /static
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "svg")
os.makedirs("static/images", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Dynamic Database Loader with Automatic Category-to-Image Connection
def load_store_data() -> dict:
    """
    Loads products.json fresh on demand and automatically links each product
    to its image inside static/images/<category>/<id>.(jpg|png|webp)
    without human intervention.
    """
    with open("products.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure category image folders exist
    for cat in data.get("categories", []):
        os.makedirs(os.path.join("static", "images", cat["id"]), exist_ok=True)

    # Automatically resolve product image URLs by category and product ID
    for item in data.get("items", []):
        if item.get("image"):
            continue
        cat_id = item.get("category", "misc")
        prod_id = item.get("id", "")
        resolved_url = None

        for ext in IMAGE_EXTENSIONS:
            candidate_path = os.path.join("static", "images", cat_id, f"{prod_id}.{ext}")
            if os.path.exists(candidate_path):
                resolved_url = f"/static/images/{cat_id}/{prod_id}.{ext}"
                break

        # Default path if image file is added later (frontend falls back to category icon if 404)
        item["image"] = resolved_url or f"/static/images/{cat_id}/{prod_id}.jpg"

    return data

# 4. Multi-User Chat Session Memory Store
# Structure: { session_id: { "updated_at": float, "history": List[types.Content] } }
CHAT_SESSIONS: Dict[str, dict] = {}
SESSION_TTL_SECONDS = 60 * 60 * 24  # 24 hours

def cleanup_expired_sessions():
    now = time.time()
    expired = [
        sid for sid, state in CHAT_SESSIONS.items()
        if now - state.get("updated_at", now) > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del CHAT_SESSIONS[sid]

def get_user_session_history(session_id: str) -> List[types.Content]:
    cleanup_expired_sessions()
    if session_id not in CHAT_SESSIONS:
        CHAT_SESSIONS[session_id] = {"updated_at": time.time(), "history": []}
    CHAT_SESSIONS[session_id]["updated_at"] = time.time()
    return CHAT_SESSIONS[session_id]["history"]

# 5. Initialize Gemini Client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# 6. Pydantic Schemas for Structured AI Output
class RecommendedItem(BaseModel):
    id: str = Field(description="Exact product 'id' from the store inventory (e.g., 'uno', 'l298n', 'hcsr04').")
    qty: int = Field(description="Recommended quantity needed for the user's project (minimum 1).")
    role_in_project: str = Field(description="Short explanation of why this specific part is needed in the project.")

class AdvisorResponse(BaseModel):
    reply: str = Field(
        description="Friendly, encouraging engineering explanation answering the user's question and explaining how the parts work together."
    )
    recommended_items: List[RecommendedItem] = Field(
        default_factory=list,
        description="List of matching products from our store inventory that the user needs."
    )
    missing_items: List[str] = Field(
        default_factory=list,
        description="General hardware parts required for the project that are NOT currently in our store inventory (if any)."
    )
    follow_up_suggestions: List[str] = Field(
        default_factory=list,
        description="2 to 3 short clickable follow-up questions the user might want to ask next."
    )

class ChatRequest(BaseModel):
    session_id: str = Field(description="Unique client session UUID to isolate each user's chat.")
    message: str

class ResetSessionRequest(BaseModel):
    session_id: str
def build_system_prompt(store_data: dict) -> str:
    # Strip images to save tokens and keep prompt execution fast
    clean_items = [
        {k: v for k, v in item.items() if k != "image"}
        for item in store_data.get("items", [])
    ]
    catalog_text = json.dumps(clean_items, ensure_ascii=False, indent=2)

    return f"""# 1. ROLE & PERSONA
أنت "مهندس متجر الكترونياتك" (Smart Store Engineer)، خبير الإلكترونيات والروبوتات في متجر '{store_data["store_name"]}' في الأردن.
الجمهور: طلاب الهندسة، الهواة والمبتدئون.
النبرة: هندسية احترافية، دقيقة، واثقة، ومباشرة بدون مقدمات إنشائية أو ترحيبات مكررة.
اللغة: اكتب باللغة العربية الواضحة، واستخدم المصطلحات التقنية بالإنجليزية (مثل: Microcontroller, Logic Level, Motor Driver, Voltage Divider).
العملة الرسمية: جميع الأسعار بالدينار الأردني ({store_data["currency"]}).

# 2. STRICT INVENTORY GROUNDING & SEARCH RULES
- لديك كتالوج المتجر الكامل في قسم [STORE INVENTORY] أدناه وهو مصدر الحقيقة الوحيد.
- قاعدة صارمة: يُمنع منعاً باتاً اختراع أي قطعة أو سعر غير موجود في الكتالوج.
- افحص [STORE INVENTORY] بدقة قبل تصنيف أي قطعة على أنها ناقصة:
  * المقاومات (Resistors) متوفرة بالمتجر بمعرفات (`res1`, `res25`). إذا احتاج المشروع مقسم جهد (Voltage Divider) أو مقاومات ليد، أضف المقاومة المتوفرة فوراً إلى `recommended_items` ولا تضعها أبداً في `missing_items`.
  * الأسلاك والبردبورد والبطاريات متوفرة، تحقق من وجودها واقترحها دوماً من الكتالوج.
- قائمة `recommended_items`: مصفوفة كائنات بالمعرف الحقيقي فقط من الكتالوج: [{{"id": "...", "qty": 1, "role_in_project": "..."}}].
- قائمة `missing_items`: مخصصة فقط وحصرياً للقطع الضرورية للمشروع التي لا يوجد لها أي أثر أو بديل وظيفي داخل الكتالوج (مثال: محرك سيرفو SG90 إذا لم يكن مدرجاً، أو بطاريات 18650 إذا كان المتوفر حوامل البطاريات فقط).

# 3. MANDATORY VISUAL FORMATTING (قواعد تنسيق حقل `reply`)
- يُمنع منعاً باتاً كتابة الرد كفقرة واحدة متصلة أو متراكمة!
- قسّم ردك دائماً إلى فقرات قصيرة ومنفصلة بأسطر فارغة مزدوجة.
- استخدم عناوين فرعية قصيرة تبدأ بـ `### ` لكل فكرة أساسية (مثال: `### 1. مبدأ العمل والتوصيل`، `### 2. جدول المنافذ`، `### 3. القطع التكميلية`).
- استخدم القوائم النقطية التي تبدأ بـ `- ` مع كتابة أسماء القطع والمصطلحات بخط عريض `**اسم القطعة**`.
- عند شرح التوصيل، استخدم جدول Markdown واضح:
  | الطرف في الحساس | المنفذ في البورد | الوظيفة / ملاحظات الجهد |
  |---|---|---|
- التنبيهات الكهربائية وحماية المداخل (مثل فرق الجهد 5V و 3.3V) توضع في سطر مستقل تماماً يبدأ بـ:
  ⚠️ **تنبيه هندسي:** [الشرح والحل]
- يُمنع منعاً باتاً كتابة أو توفير أي أكواد برمجية (No Code Generation). اشرح التوصيل والمنطق فقط.

# 4. INTENT ROUTING
1. بناء مشروع (Project Build):
   - قسّم المنظومة هندسياً (العقل المتحكم، المشغلات/الدرايفرات، الحساسات، التغذية).
   - اعرض جدول توصيل الأطراف فقط (لا تقم بإعطاء أي كود برمجي).
2. المقارنة أو الشرح (Comparison / Technical Q&A):
   - افصل بين القطع بعناوين `### ` مع إيضاح نقاط القوة والمدى التشغيلي لكل منها.
3. التوافقية وجهود التشغيل (Logic Levels & Power):
   - نبّه فوراً لأي تعارض في الجهود مع اقتراح دائرة الحماية المناسبة كالمقاومات المتوفرة بالمتجر.
4. استكشاف الأعطال (Troubleshoot):
   - ابدأ بالأسباب الأكثر احتمالاً (توصيل الأرضي المشترك Common GND، كفاية التيار، صحة المنافذ).
5. الأسئلة العامة أو الخارجة عن الاختصاص (Off-Topic):
   - اعتذر بلباقة في جملة واحدة وأعد التوجيه لمكونات ومشاريع الإلكترونيات.

# 5. OUTPUT FORMAT
التزم بإرجاع كائن JSON مطابق تماماً للـ Schema التالية:
- `reply`: الشرح الفني الغني المنسق بالعناوين `### ` والجداول والتنبيهات `⚠️` (بدون سرد الأسعار نصياً لأن واجهة المتجر تبني كروت السلة تلقائياً).
- `recommended_items`: مصفوفة القطع المتوفرة بالمتجر.
- `missing_items`: مصفوفة بأسماء القطع الخارجية الناقصة فعلياً فقط.
- `follow_up_suggestions`: من 2 إلى 3 أسئلة تالية مختصرة ومفيدة للمستخدم.

[STORE INVENTORY]
{catalog_text}
"""
# 7. Routes
@app.get("/")
def serve_home():
    return FileResponse("index.html")

@app.get("/api/products")
def get_products():
    """Returns the live store catalog with automatically resolved category image URLs."""
    return load_store_data()

@app.post("/api/chat/reset")
def reset_chat_session(req: ResetSessionRequest):
    """Clears the conversation history for a specific user's session_id."""
    if req.session_id in CHAT_SESSIONS:
        del CHAT_SESSIONS[req.session_id]
    return {"status": "cleared", "session_id": req.session_id}

@app.post("/api/chat", response_model=AdvisorResponse)
def chat_with_advisor(req: ChatRequest):
    if not client:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable is not set on the server."
        )

    store_data = load_store_data()
    user_history = get_user_session_history(req.session_id)

    # Keep the last 8 messages (4 turns) for this specific user session
    recent_contents = list(user_history[-8:])
    user_msg_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=req.message)]
    )
    recent_contents.append(user_msg_content)

    # UPDATED: Only try models that are fast and currently active to prevent timeouts
    models_to_try = [
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-flash-latest",
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=recent_contents,
                config=types.GenerateContentConfig(
                    system_instruction=build_system_prompt(store_data),
                    temperature=0.3,
                    response_mime_type="application/json",
                    response_schema=AdvisorResponse,
                ),
            )
            parsed = AdvisorResponse.model_validate_json(response.text)

            # Save turn to this user's isolated session history
            user_history.append(user_msg_content)
            user_history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=parsed.reply)]
                )
            )
            return parsed
        except Exception as e:
            last_error = e
            print(f"⚠️ Model '{model_name}' failed: {e}")
            continue

    raise HTTPException(status_code=500, detail=str(last_error))