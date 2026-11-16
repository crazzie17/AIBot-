# @title Step 2: Choose provider + paste API key
PROVIDER = "nvidia"  # @param ["openrouter","groq", "nvidia", "gemini"]
API_KEY = "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"  # @param {type:"string"}
MODEL = "openai/gpt-oss-20b" # @param {type:"string"}

# Each provider = an address (base_url) + a model name.
# Trainer tip: to change a model, just edit the "model" text below.
PROVIDERS = {
    "groq":       {"base_url": "https://api.groq.com/openai/v1"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1"},   # any model ending in ":free" works
    "nvidia":     {"base_url": "https://integrate.api.nvidia.com/v1"},
    "gemini":     {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"}
}

from openai import OpenAI

API_KEY = API_KEY.strip()
if API_KEY == "":
    raise ValueError("✋ Please paste your API key in the API_KEY box and run this cell again.")


client = OpenAI(api_key=API_KEY, base_url=PROVIDERS[PROVIDER]["base_url"])

print("✅ Settings saved")
print("   Provider :", PROVIDER)
print("   Model    :", MODEL)



def ask_llm(messages, temperature=0.7):
    """Send messages to the AI and return its reply as text."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,   # 0 = focused, 1 = creative
        )
        reply = response.choices[0].message.content
        return reply if reply else "(Empty reply — please run the cell again.)"

    except Exception as error:
        details = str(error)
        if "401" in details or "403" in details or "API key" in details:
            tip = "Your API key is wrong or inactive. Copy it again and re-run Step 2."
        elif "429" in details:
            tip = "Free limit reached for now. Wait 1 minute and try again."
        elif "404" in details:
            tip = "Model not found. Check the model name in Step 2."
        else:
            tip = "Something went wrong. Check your key / internet, or try another PROVIDER."
        return f"❌ {tip}\n   (Details: {details[:200]})"


# Quick test 👇
test = ask_llm([{"role": "user", "content": "Say hello to the students of the India Book of Records event in one line."}])
print("🤖", test)




# @title A1: Design your chatbot
BOT_NAME = ""  # @param {type:"string"}
ROLE = ""  # @param {type:"string"}
TONE = ""  # @param {type:"string"}
LANGUAGE = ""  # @param {type:"string"}
ANSWER_LENGTH = ""  # @param {type:"string"}

# 👇 The PROMPT TEMPLATE — words inside { } are blanks that get filled in.
PROMPT_TEMPLATE = """
You are {bot_name}, {role}.
Your tone is {tone}.
Always reply in {language}.
Keep every answer to {answer_length}.
If someone asks about something outside your role, politely say what you can help with instead.
"""

SYSTEM_PROMPT = PROMPT_TEMPLATE.format(
    bot_name=BOT_NAME,
    role=ROLE,
    tone=TONE,
    language=LANGUAGE,
    answer_length=ANSWER_LENGTH,
)

print("✅ Your chatbot's system prompt is:")
print(SYSTEM_PROMPT)


# @title A2: Chat with your bot (run again for the next message)
QUESTION = "''"  # @param {type:"string"}

if "chat_history" not in globals() or chat_history[0]["content"] != SYSTEM_PROMPT:
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

chat_history.append({"role": "user", "content": QUESTION})
reply = ask_llm(chat_history[:1] + chat_history[-10:])

if reply.startswith("❌"):
    chat_history.pop()
    print(reply)
else:
    chat_history.append({"role": "assistant", "content": reply})
    print(f"💬 Chatting with: {BOT_NAME}\n")
    for msg in chat_history[1:]:
        if msg["role"] == "user":
            print(f"🧑 You: {msg['content']}")
        else:
            print(f"🤖 {BOT_NAME}: {msg['content']}\n")




import requests, faiss, numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---- Default settings (trainers can tweak; students need not touch) ----
CHUNK_SIZE    = 800    # characters per chunk
CHUNK_OVERLAP = 150    # characters repeated between chunks so sentences aren't lost
TOP_K         = 3      # how many chunks are given to the AI for each question

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant who answers ONLY using the context below.
If the answer is not in the context, reply: "Sorry, I could not find this in the document."
Answer in 3 to 4 simple sentences.

Context:
{context}

Question: {question}

Answer:
"""

# ---- STEP 1: Download the chosen PDF ----
category, pdf_file = category_box.value, pdf_box.value
if category == RANDOM:
    category = random.choice(list(CATALOG))
if pdf_file == RANDOM:
    pdf_file = random.choice(CATALOG[category])
PDF_TITLE = pretty(pdf_file)
link = f"https://raw.githubusercontent.com/{REPO}/main/{PDF_FOLDER}/{category}/{pdf_file}"
response = requests.get(link, timeout=60)
if response.status_code != 200:
    raise ValueError(f"❌ Could not download the PDF (error {response.status_code}). Re-run B1 and try again.")
with open("my_document.pdf", "wb") as f:
    f.write(response.content)
print(f"1️⃣  PDF downloaded: {PDF_TITLE}")
print(f"📖 Read the PDF here: https://github.com/{REPO}/blob/main/{PDF_FOLDER}/{category}/{pdf_file}")

# ---- STEP 2: Read the text, page by page ----
reader = PdfReader("my_document.pdf")
pages = []
for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    if text.strip():
        pages.append({"page": page_number, "text": text})
if len(pages) == 0:
    raise ValueError("❌ No text found. This PDF may be a scanned image. Try another PDF.")
print(f"2️⃣  Read {len(pages)} pages")

# ---- STEP 3: Cut the text into small overlapping chunks ----
chunks = []
for p in pages:
    text = " ".join(p["text"].split())                               # clean extra spaces / new lines
    start = 0
    while start < len(text):
        chunks.append({"page": p["page"], "text": text[start : start + CHUNK_SIZE]})
        start = start + CHUNK_SIZE - CHUNK_OVERLAP
print(f"3️⃣  Created {len(chunks)} chunks")

# ---- STEP 4: Turn each chunk into 384 numbers (embeddings) ----
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
vectors = np.array(vectors, dtype="float32")
print("4️⃣  Embeddings created")

# ---- STEP 5: Store them in FAISS (a fast search engine for numbers) ----
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
print(f"5️⃣  Stored {index.ntotal} chunks in FAISS")

# ---- The RAG function: retrieve -> augment -> generate ----
def rag_ask(question):
    q_vector = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_vector, TOP_K)                       # 1. RETRIEVE similar chunks
    found = [chunks[i] for i in ids[0] if i != -1]
    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in found)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)        # 2. AUGMENT the prompt
    answer = ask_llm([{"role": "user", "content": prompt}], temperature=0.2)       # 3. GENERATE the answer
    return answer, sorted(set(c["page"] for c in found))

print("\n✅ RAG engine {} is ready! Now run B3 and ask questions.".format(PDF_TITLE))



# @title B3: Ask your PDF bot (run again for the next question)
RAG_BOT_NAME = ""  # @param {type:"string"}
PDF_QUESTION = ""  # @param {type:"string"}

bot_name = RAG_BOT_NAME.strip() or PDF_TITLE
answer, pages_used = rag_ask(PDF_QUESTION)

print(f"📄 {bot_name}  ·  PDF: {PDF_TITLE}\n")
print(f"🧑 You: {PDF_QUESTION}\n")
print(f"🤖 {bot_name}: {answer}")
print(f"   📄 Source pages: {pages_used}")



