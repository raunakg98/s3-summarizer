import json, os, urllib.request, boto3, urllib.parse

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")

# ── model mapping ───────────────────────────────────────────────
MODELS = {
    "haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "titan": "amazon.titan-text-lite-v1"
}
DEFAULT_MODEL = MODELS["haiku"]            # <- default

# ── helpers ------------------------------------------------------
def _invoke_bedrock(prompt: str, model_id: str) -> str:
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {"maxTokenCount": 256, "temperature": 0.2}
    })
    resp = bedrock.invoke_model(modelId=model_id, body=body)
    return json.loads(resp["body"].read())["results"][0]["outputText"]

def _summarise_chunk(text: str, model: str) -> str:
    prompt = (
        "You are a helpful assistant.\n"
        "Summarize the following passage in exactly five short bullet points.\n"
        "Each point should describe a key event or idea.\n"
        "Only return five bullets. Do not continue the story.\n\n"
        f"{text}"
    )
    return _invoke_bedrock(prompt, model)

def summarise(text: str, model: str) -> str:
    chunks = [text[i:i+4800] for i in range(0, len(text), 4800)][:4]
    bullets = [_summarise_chunk(c, model) for c in chunks]
    return _summarise_chunk(" ".join(bullets), model)

# ---------- helper to add CORS headers -----------------------------
def _cors(resp: dict) -> dict:
    resp.setdefault("headers", {})
    resp["headers"]["Access-Control-Allow-Origin"]  = "*"
    resp["headers"]["Access-Control-Allow-Headers"] = "*"
    return resp

# ── Lambda entrypoint -------------------------------------------
def lambda_handler(event, _):
    model_key = json.loads(event.get("body") or "{}").get("model") if "body" in event else event.get("model")
    model_id  = MODELS.get((model_key or "haiku").lower(), DEFAULT_MODEL)

    # --- S3 trigger (unchanged) ---
    #  (use DEFAULT_MODEL for automated pipeline)
    if "Records" in event and event["Records"][0]["eventSource"].startswith("aws:s3"):
        # ... same as before ...
        summary = summarise(text, DEFAULT_MODEL)
        # ...

    # --- /summarise-text -----------
    if event.get("rawPath") == "/summarise-text":
        text = json.loads(event["body"]).get("text","")
        return _cors({"statusCode":200,"body":json.dumps({"summary": summarise(text, model_id)})})

    # --- /summarise -----------------
    if event.get("rawPath") == "/summarise":
        url = json.loads(event["body"]).get("url","")
        text = urllib.request.urlopen(url,timeout=60).read().decode("utf-8","ignore")
        return _cors({"statusCode":200,"body":json.dumps({"summary": summarise(text, model_id)})})

    # --- CLI direct invoke ----------
    if "text" in event or "url" in event:
        raw = event.get("text") or urllib.request.urlopen(event["url"],timeout=60).read().decode("utf-8","ignore")
        return _cors({"statusCode":200,"body":json.dumps({"summary": summarise(raw, model_id)})})

    return _cors({"statusCode":404,"body":"Not found"})

