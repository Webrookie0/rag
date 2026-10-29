"""Optional local Ollama chat adapter, using only the Python standard library.

No service is started and no model is downloaded by this module.
Set OLLAMA_MODEL to a model already installed in your local Ollama server.
"""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen


def chat(system: str, user: str, json_output: bool = False) -> str:
    body = {
        "model": os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": 0},
    }
    if json_output:
        body["format"] = "json"
    request = Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except (URLError, TimeoutError) as exc:
        raise RuntimeError("Ollama request failed. Start Ollama and set OLLAMA_MODEL to an installed model.") from exc
    content = result.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned no answer text")
    return content
