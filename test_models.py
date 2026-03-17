import urllib.request, urllib.error, json, sys, time

api_key = sys.argv[1]
test_prompt = "Translate 'hello' to French. Reply with one word only."

models = [
    ("v1beta", "gemini-2.5-flash"),
    ("v1beta", "gemini-2.5-pro"),
    ("v1",     "gemini-2.0-flash"),
    ("v1",     "gemini-2.0-flash-001"),
    ("v1",     "gemini-2.0-flash-lite-001"),
    ("v1beta", "gemini-2.0-flash"),
    ("v1beta", "gemini-2.0-flash-001"),
    ("v1beta", "gemini-2.5-flash-lite"),
    ("v1beta", "gemini-flash-latest"),
]

payload = json.dumps({
    "contents": [{"parts": [{"text": test_prompt}]}],
    "generationConfig": {"temperature": 0.0},
}).encode()

print(f"{'version':<8} {'model':<30} {'result'}")
print("-" * 65)
for version, model in models:
    url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent?key={api_key}"
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        print(f"{version:<8} {model:<30} OK: {text[:30]}")
    except urllib.error.HTTPError as e:
        print(f"{version:<8} {model:<30} HTTP {e.code}")
    except Exception as e:
        print(f"{version:<8} {model:<30} ERR: {str(e)[:30]}")
    time.sleep(1)
