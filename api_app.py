# api_app.py
# Full server: receives tasks, calls OpenRouter/gpt-4.1-nano to generate files,
# handles attachments, deploys to GitHub Pages, and supports robust captcha-solver UX.

import os
import re
import json
import base64
import shutil
import asyncio
import logging
import sys
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

import httpx
import git
import psutil
from openai import OpenAI

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# -------------------------
# Settings
# -------------------------
class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field("", env="OPENAI_API_KEY")
    GITHUB_TOKEN: str = Field("", env="GITHUB_TOKEN")
    GITHUB_USERNAME: str = Field("", env="GITHUB_USERNAME")
    STUDENT_SECRET: str = Field("", env="STUDENT_SECRET")
    LOG_FILE_PATH: str = Field("logs/app.log", env="LOG_FILE_PATH")
    MAX_CONCURRENT_TASKS: int = Field(2, env="MAX_CONCURRENT_TASKS")
    KEEP_ALIVE_INTERVAL_SECONDS: int = Field(30, env="KEEP_ALIVE_INTERVAL_SECONDS")
    GITHUB_API_BASE: str = Field("https://api.github.com", env="GITHUB_API_BASE")
    GITHUB_PAGES_BASE: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
if not settings.GITHUB_PAGES_BASE:
    settings.GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"

# -------------------------
# Logging
# -------------------------
os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
logger = logging.getLogger("task_receiver")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
file_handler = logging.FileHandler(settings.LOG_FILE_PATH, mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers = []
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False

def flush_logs():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass

# -------------------------
# Models
# -------------------------
class Attachment(BaseModel):
    name: str
    url: str

class TaskRequest(BaseModel):
    task: str
    email: str
    round: int
    brief: str
    evaluation_url: str
    nonce: str
    secret: str
    attachments: List[Attachment] = []

# -------------------------
# App & globals
# -------------------------
app = FastAPI(title="AI Web App Builder", description="Generate & deploy single-file web apps via OpenRouter")
background_tasks_list: List[asyncio.Task] = []
task_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
last_received_task: Optional[dict] = None

OPENAI_MODEL = "openai/gpt-4.1-nano"
OPENAI_BASE_URL = "https://aipipe.org/openrouter/v1"
OPENAI_API_KEY = settings.OPENAI_API_KEY

GENERATED_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_files",
        "description": "Return JSON with index.html, README.md, LICENSE",
        "parameters": {
            "type": "object",
            "properties": {
                "index.html": {"type": "string"},
                "README.md": {"type": "string"},
                "LICENSE": {"type": "string"}
            },
            "required": ["index.html", "README.md", "LICENSE"],
            "additionalProperties": False
        }
    }
}

# -------------------------
# Helpers
# -------------------------
def verify_secret(secret_from_request: str) -> bool:
    return secret_from_request == settings.STUDENT_SECRET

async def fetch_url_as_base64(url: str, timeout: int = 20) -> Optional[Dict[str,str]]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "application/octet-stream")
            b64 = base64.b64encode(r.content).decode("utf-8")
            return {"mime": content_type, "b64": b64}
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

# Process attachments: return either image inlineData or text block
async def process_attachment_for_llm(attachment_url: str) -> Optional[dict]:
    if not attachment_url or not attachment_url.startswith(("data:", "http")):
        logger.warning(f"Invalid attachment URL: {attachment_url}")
        return None
    try:
        if attachment_url.startswith("data:"):
            match = re.search(r"data:(?P<mime>[^;]+);base64,(?P<data>.*)", attachment_url, re.IGNORECASE)
            if not match:
                return None
            mime = match.group("mime")
            b64 = match.group("data")
            if mime.startswith("image/"):
                return {"inlineData": {"data": b64, "mimeType": mime}}
            else:
                decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
                if len(decoded) > 20000:
                    decoded = decoded[:20000] + "\n\n...TRUNCATED..."
                return {"type": "text", "text": f"ATTACHMENT ({mime}):\n{decoded}"}
        else:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(attachment_url)
                resp.raise_for_status()
                mime = resp.headers.get("Content-Type", "application/octet-stream")
                b64 = base64.b64encode(resp.content).decode("utf-8")
                if mime and mime.startswith("image/"):
                    return {"inlineData": {"data": b64, "mimeType": mime}}
                lower = attachment_url.lower()
                if mime in ("text/csv", "application/json", "text/plain") or lower.endswith((".csv", ".json", ".txt")):
                    decoded = resp.content.decode("utf-8", errors="ignore")
                    if len(decoded) > 20000:
                        decoded = decoded[:20000] + "\n\n...TRUNCATED..."
                    return {"type":"text", "text": f"ATTACHMENT ({mime}):\n{decoded}"}
                return None
    except Exception as e:
        logger.exception(f"Error processing attachment {attachment_url}: {e}")
        return None

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

async def save_generated_files_locally(task_id: str, files: dict) -> str:
    base_dir = os.path.join(os.getcwd(), "generated_tasks")
    task_dir = os.path.join(base_dir, task_id)
    safe_makedirs(task_dir)
    logger.info(f"Saving generated files to {task_dir}")
    for fname, content in files.items():
        p = os.path.join(task_dir, fname)
        await asyncio.to_thread(lambda p, c: open(p, "w", encoding="utf-8").write(c), p, content)
        logger.info(f"  saved {fname}")
    flush_logs()
    return task_dir

async def save_attachments_locally(task_dir: str, attachments: List[Attachment]) -> List[str]:
    saved = []
    async with httpx.AsyncClient(timeout=30) as client:
        for att in attachments:
            try:
                if att.url.startswith("data:"):
                    m = re.search(r"base64,(.*)", att.url, re.IGNORECASE)
                    if not m:
                        continue
                    data = base64.b64decode(m.group(1))
                else:
                    resp = await client.get(att.url)
                    resp.raise_for_status()
                    data = resp.content
                p = os.path.join(task_dir, att.name)
                await asyncio.to_thread(lambda p, d: open(p, "wb").write(d), p, data)
                saved.append(att.name)
                logger.info(f"Saved attachment {att.name}")
            except Exception as e:
                logger.exception(f"Failed to save attachment {att.name}: {e}")
    flush_logs()
    return saved

def remove_local_path(path: str):
    if not os.path.exists(path):
        return
    logger.info(f"Removing local path {path}")
    def _try_rmtree(p):
        try:
            shutil.rmtree(p)
            return True
        except Exception as e:
            logger.warning(f"rmtree attempt failed: {e}")
            return False
    for i in range(6):
        if _try_rmtree(path):
            return True
        # try to terminate processes holding files under path (Windows)
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    for f in proc.open_files():
                        try:
                            if os.path.commonpath([os.path.abspath(path), os.path.abspath(f.path)]) == os.path.abspath(path):
                                logger.warning(f"Terminating process {proc.pid} ({proc.name()}) holding {f.path}")
                                try: proc.terminate()
                                except Exception: pass
                        except Exception:
                            continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        time.sleep(1.0)
    logger.error(f"Failed to remove {path}")
    return False

# Git helpers
async def setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int) -> git.Repo:
    gh_user = settings.GITHUB_USERNAME
    gh_token = settings.GITHUB_TOKEN
    headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"}
    should_clone = (round_index > 1)
    creation_succeeded = False
    if round_index == 1:
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                payload = {"name": repo_name, "private": False, "auto_init": True}
                r = await client.post(f"{settings.GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                if r.status_code == 201:
                    creation_succeeded = True
                elif r.status_code == 422:
                    msg = r.json().get("message", "")
                    if "already exists" in msg:
                        should_clone = True
                    else:
                        r.raise_for_status()
                else:
                    r.raise_for_status()
            except Exception as e:
                logger.exception(f"Repo create error: {e}")
                raise
    if should_clone or (round_index > 1 and not creation_succeeded):
        try:
            repo = await asyncio.to_thread(git.Repo.clone_from, repo_url_auth, local_path)
            logger.info("Cloned repo")
            return repo
        except Exception as e:
            logger.exception(f"Clone failed: {e}")
            raise
    else:
        repo = git.Repo.init(local_path)
        repo.create_remote('origin', repo_url_auth)
        logger.info("Initialized local repo")
        return repo

async def commit_and_publish(repo: git.Repo, task_id: str, round_index: int, repo_name: str) -> dict:
    gh_user = settings.GITHUB_USERNAME
    gh_token = settings.GITHUB_TOKEN
    repo_url_http = f"https://github.com/{gh_user}/{repo_name}"
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            await asyncio.to_thread(repo.git.add, A=True)
            msg = f"Task {task_id} - Round {round_index}"
            commit = await asyncio.to_thread(lambda m: repo.index.commit(m), msg)
            commit_sha = getattr(commit, "hexsha", "")
            await asyncio.to_thread(lambda *args: repo.git.branch(*args), '-M', 'main')
            await asyncio.to_thread(lambda r: r.git.push('--set-upstream', 'origin', 'main', force=True), repo)
            # configure pages
            await asyncio.sleep(2)
            pages_api = f"{settings.GITHUB_API_BASE}/repos/{gh_user}/{repo_name}/pages"
            payload = {"source": {"branch": "main", "path": "/"}}
            for attempt in range(5):
                try:
                    resp = await client.get(pages_api, headers={"Authorization": f"token {gh_token}"})
                    if resp.status_code == 200:
                        await client.put(pages_api, json=payload, headers={"Authorization": f"token {gh_token}"})
                    else:
                        await client.post(pages_api, json=payload, headers={"Authorization": f"token {gh_token}"})
                    break
                except httpx.HTTPStatusError as e:
                    text = e.response.text.lower() if e.response and e.response.text else ""
                    if e.response.status_code == 422 and "main branch must exist" in text and attempt < 4:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
            await asyncio.sleep(5)
            pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/"
            return {"repo_url": repo_url_http, "commit_sha": commit_sha, "pages_url": pages_url}
        except Exception as e:
            logger.exception(f"Commit/publish failed: {e}")
            raise

# LLM wrapper with updated system prompt for captcha behavior
async def call_llm_for_code(prompt: str, task_id: str, attachment_parts: List[dict]) -> dict:
    logger.info(f"[LLM] Generating code for {task_id}")
    system_prompt = (
        "You are an expert full-stack web engineer. MUST return exactly one function call to 'generate_files' "
        "with keys: index.html, README.md, LICENSE (complete file contents). No extra text.\n\n"
        "IMPORTANT: If the brief or user asks for handling a ?url=... parameter or a remote image URL, implement the following behavior in the generated index.html:\n"
        " - Detect a URL parameter named 'url' (e.g., ?url=https://.../image.png). If present, attempt to load that image into an <img> element with crossOrigin='anonymous'.\n"
        " - Use a robust client-side OCR fallback using Tesseract.js (via CDN). Attempt OCR on the loaded image and store the OCRed text.\n"
        " - Provide an input box for the user to type the captcha text and a Submit button. On submit, compare the user's input to the OCRed text (case-insensitive) and show success/failure.\n"
        " - If loading the remote image fails (CORS, network, 404) or OCR fails, default to an attached sample image that is stored in the project root (e.g., './sample.png').\n"
        " - Embed the attached sample image in the generated files by referencing the attachment filename AND include a base64 inline fallback so the page always works.\n"
        " - Avoid server-side calls: app must work client-side in the browser. Handle CORS gracefully: if the remote image is tainted, display an informative message and fall back to sample.\n"
        " - Include minimal accessible UI: image preview, OCR status, input box, submit button, result message, and a 'Try URL' area showing the parsed ?url value.\n\n"
        "ROUND RULES:\n"
        " - Round 1: Create full single-file Tailwind index.html implementing above behavior if requested; include README.md and MIT LICENSE.\n"
        " - Round 2+: Make minimal precise edits to the previously generated files; preserve layout/style.\n\n"
        "FILES:\n"
        " index.html: single-file HTML using Tailwind CDN + vanilla JS + Tesseract.js via CDN. Must reference any attached image file by filename in the root (./<name>) and include base64 fallback.\n"
        " README.md: describe the app, mention attachment usage, and Live Demo link.\n"
        " LICENSE: MIT with [year] and [author].\n"
        "Output exactly the JSON for generate_files and nothing else.\n"
    )

    # Build user content: prompt + attachments meta
    user_content = [{"type":"text","text":prompt}]
    # Append attachment parts (images as inlineData and text parts)
    for part in attachment_parts:
        if part.get("inlineData"):
            mime = part["inlineData"].get("mimeType","application/octet-stream")
            b64 = part["inlineData"].get("data","")
            user_content.append({"type":"image_url","image_url":{"url":f"data:{mime};base64,{b64}"}})
        elif part.get("type") == "text" and part.get("text"):
            user_content.append({"type":"text","text":part["text"]})

    try:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, http_client=httpx.Client(timeout=90.0))
    except Exception as e:
        logger.error(f"OpenRouter client init failed: {e}")
        raise

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_content}],
                tools=[GENERATED_CODE_TOOL],
                tool_choice={"type":"function","function":{"name":"generate_files"}},
                temperature=0.0
            )
            response_message = response.choices[0].message
            tool_calls = getattr(response_message, "tool_calls", None) or []
            if tool_calls and tool_calls[0].function.name == "generate_files":
                json_text = tool_calls[0].function.arguments
                generated = json.loads(json_text)
                # Basic validation
                for k in ("index.html","README.md","LICENSE"):
                    if k not in generated:
                        raise ValueError(f"Missing {k} in LLM output")
                return generated
            else:
                raise ValueError("Model did not return expected tool call")
        except Exception as e:
            logger.warning(f"[LLM] Attempt {attempt+1} error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    raise Exception("LLM generation failed after retries")

# Notify
async def notify_evaluation_server(evaluation_url: str, email: str, task_id: str, round_index: int, nonce: str, repo_url: str, commit_sha: str, pages_url: str) -> bool:
    if not evaluation_url or "example.com" in evaluation_url or evaluation_url.strip()=="":
        logger.warning("Skipping notify due to invalid URL")
        return False
    payload = {"email":email,"task":task_id,"round":round_index,"nonce":nonce,"repo_url":repo_url,"commit_sha":commit_sha,"pages_url":pages_url}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(evaluation_url, json=payload)
                r.raise_for_status()
                logger.info("Notify succeeded")
                return True
        except Exception as e:
            logger.warning(f"Notify attempt {attempt+1} failed: {e}")
            if attempt < max_retries-1:
                await asyncio.sleep(2 ** attempt)
    logger.error("Notify failed after retries")
    return False

# Main orchestration
async def generate_files_and_deploy(task_data: TaskRequest):
    acquired = False
    try:
        await task_semaphore.acquire()
        acquired = True
        logger.info(f"Start task {task_data.task} round {task_data.round}")
        task_id = task_data.task
        round_index = task_data.round
        brief = task_data.brief
        attachments = task_data.attachments or []

        repo_name = task_id.replace(" ","-").lower()
        gh_user = settings.GITHUB_USERNAME
        gh_token = settings.GITHUB_TOKEN
        repo_url_auth = f"https://{gh_user}:{gh_token}@github.com/{gh_user}/{repo_name}.git"
        repo_url_http = f"https://github.com/{gh_user}/{repo_name}"

        base_dir = os.path.join(os.getcwd(),"generated_tasks")
        local_path = os.path.join(base_dir, task_id)

        if os.path.exists(local_path):
            try:
                await asyncio.to_thread(remove_local_path, local_path)
            except Exception as e:
                logger.exception(f"Cleanup failed: {e}")
                raise

        safe_makedirs(local_path)

        repo = await setup_local_repo(local_path, repo_name, repo_url_auth, repo_url_http, round_index)

        # Process attachments and build metadata summary for LLM
        attachment_parts: List[dict] = []
        attachment_meta_lines = []
        for att in attachments:
            # attempt to fetch and convert to inlineData or text
            part = await process_attachment_for_llm(att.url)
            if part:
                attachment_parts.append(part)
            # build meta
            lower = att.name.lower()
            kind = "image" if lower.endswith((".png",".jpg",".jpeg",".gif")) else "file"
            attachment_meta_lines.append(f"{att.name} ({kind}) - url: {att.url}")
        if attachment_meta_lines:
            meta_text = "ATTACHMENTS METADATA:\n" + "\n".join(attachment_meta_lines)
            # include metadata as text block
            attachment_parts.append({"type":"text","text":meta_text})

        # Build user prompt: round aware and mention attachments by name
        if round_index > 1:
            llm_prompt = (
                f"UPDATE (ROUND {round_index}): Make minimal edits to existing project to implement: {brief}. "
                "Preserve structure and style. Provide full replacement content for index.html, README.md, LICENSE."
            )
        else:
            llm_prompt = (
                f"CREATE (ROUND {round_index}): Build a complete single-file responsive Tailwind web app for: {brief}. "
                "Provide index.html, README.md, and MIT LICENSE. If attachments are included, incorporate them."
            )
        if attachment_meta_lines:
            llm_prompt += "\n\n" + "Provided attachments:\n" + "\n".join(attachment_meta_lines)

        generated_files = await call_llm_for_code(llm_prompt, task_id, attachment_parts)

        # Save generated files locally
        task_dir = await save_generated_files_locally(task_id, generated_files)

        # Save attachments into the same repo dir
        saved_names = await save_attachments_locally(task_dir, attachments)

        # Commit & push
        deployment = await commit_and_publish(repo, task_id, round_index, repo_name)
        repo_url = deployment["repo_url"]
        commit_sha = deployment["commit_sha"]
        pages_url = deployment["pages_url"]
        logger.info(f"Deployed: {pages_url}")

        # Notify evaluator (skip example.com)
        await notify_evaluation_server(task_data.evaluation_url, task_data.email, task_id, round_index, task_data.nonce, repo_url, commit_sha, pages_url)

    except Exception as e:
        logger.exception(f"Task failed: {e}")
    finally:
        if acquired:
            task_semaphore.release()
        flush_logs()

# Background callback
def _task_done_callback(task: asyncio.Task):
    try:
        exc = task.exception()
        if exc:
            logger.error(f"Background task exception: {exc}")
        else:
            logger.info("Background task finished successfully")
    except asyncio.CancelledError:
        logger.warning("Background task cancelled")
    finally:
        flush_logs()

# Endpoints
@app.post("/ready", status_code=200)
async def receive_task(task_data: TaskRequest, background_tasks: BackgroundTasks, request: Request):
    global last_received_task, background_tasks_list
    if not verify_secret(task_data.secret):
        logger.warning("Unauthorized attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")
    last_received_task = {"task": task_data.task, "round": task_data.round, "brief": (task_data.brief[:250]+"...") if len(task_data.brief)>250 else task_data.brief, "time": datetime.utcnow().isoformat()+"Z"}
    bg = asyncio.create_task(generate_files_and_deploy(task_data))
    bg.add_done_callback(_task_done_callback)
    background_tasks_list.append(bg)
    # keepalive no-op
    background_tasks.add_task(lambda: None)
    logger.info(f"Received task {task_data.task}")
    flush_logs()
    return JSONResponse(status_code=200, content={"status":"processing_scheduled","task":task_data.task})

@app.get("/")
async def root():
    return {"message":"Service running. POST /ready to submit tasks."}

@app.get("/status")
async def status():
    if last_received_task:
        return {"last_received_task": last_received_task, "running_background_tasks": len([t for t in background_tasks_list if not t.done()])}
    return {"message":"No tasks yet."}

@app.get("/health")
async def health():
    return {"status":"ok", "timestamp": datetime.utcnow().isoformat()+"Z"}

@app.get("/logs")
async def get_logs(lines: int = Query(200, ge=1, le=5000)):
    path = settings.LOG_FILE_PATH
    if not os.path.exists(path):
        return PlainTextResponse("Log file not found.", status_code=404)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            buf = bytearray()
            block = 1024
            while size > 0 and len(buf) < lines * 2000:
                read_size = min(block, size)
                f.seek(size - read_size)
                buf.extend(f.read(read_size))
                size -= read_size
            text = buf.decode(errors="ignore").splitlines()
            last_lines = "\n".join(text[-lines:])
            return PlainTextResponse(last_lines)
    except Exception as e:
        logger.exception(f"Logs read failed: {e}")
        return PlainTextResponse(f"Error: {e}", status_code=500)

# Keepalive & shutdown
@app.on_event("startup")
async def startup_event():
    async def keepalive():
        while True:
            try:
                logger.info("[KEEPALIVE] heartbeat")
                flush_logs()
            except Exception:
                pass
            await asyncio.sleep(settings.KEEP_ALIVE_INTERVAL_SECONDS)
    asyncio.create_task(keepalive())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutdown: cancel background tasks")
    for t in background_tasks_list:
        if not t.done():
            try:
                t.cancel()
            except Exception:
                pass
    await asyncio.sleep(0.5)
    flush_logs()










# # api_app.py
# """
# Automated Task Receiver & Processor
# - Round 1: Create a single-file responsive Tailwind web app (index.html), README.md, MIT LICENSE
# - Round 2+: Apply minimal changes to previous project while preserving structure/style
# - Accepts attachments (images, CSV, JSON, TXT) and includes them for LLM consumption
# - Deploys to GitHub (creates repo on round 1 or clones on round >1), pushes to main, enables GitHub Pages
# - Notifies an evaluation server (skips example.com during testing)
# - Robust cleanup with Windows file-lock handling (psutil)
# - Uses OpenRouter/OpenAI `openai/gpt-4.1-nano` model via OpenAI Python client with base_url override
# """

# import os
# import re
# import json
# import base64
# import stat
# import shutil
# import asyncio
# import logging
# import sys
# import time
# from typing import List, Optional, Dict, Any
# from datetime import datetime

# # Network / Git / LLM deps
# import httpx
# import git
# import psutil

# # OpenAI/OpenRouter client
# import openai
# from openai import OpenAI

# # Web framework
# from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
# from fastapi.responses import JSONResponse, PlainTextResponse
# from pydantic import BaseModel, Field
# from pydantic_settings import BaseSettings

# # ---------------------------------------------------------------------
# # Configuration and Settings
# # ---------------------------------------------------------------------
# class Settings(BaseSettings):
#     OPENAI_API_KEY: str = Field("", env="OPENAI_API_KEY")
#     GITHUB_TOKEN: str = Field("", env="GITHUB_TOKEN")
#     GITHUB_USERNAME: str = Field("", env="GITHUB_USERNAME")
#     STUDENT_SECRET: str = Field("", env="STUDENT_SECRET")
#     LOG_FILE_PATH: str = Field("logs/app.log", env="LOG_FILE_PATH")
#     MAX_CONCURRENT_TASKS: int = Field(2, env="MAX_CONCURRENT_TASKS")
#     KEEP_ALIVE_INTERVAL_SECONDS: int = Field(30, env="KEEP_ALIVE_INTERVAL_SECONDS")
#     GITHUB_API_BASE: str = Field("https://api.github.com", env="GITHUB_API_BASE")
#     GITHUB_PAGES_BASE: Optional[str] = None

#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"


# settings = Settings()
# if not settings.GITHUB_PAGES_BASE:
#     settings.GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"

# # ---------------------------------------------------------------------
# # Logging setup
# # ---------------------------------------------------------------------
# os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
# logger = logging.getLogger("task_receiver")
# logger.setLevel(logging.INFO)

# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

# file_handler = logging.FileHandler(settings.LOG_FILE_PATH, mode="a", encoding="utf-8")
# file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

# # Reset handlers (avoid double logging on reload)
# logger.handlers = []
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)
# logger.propagate = False

# def flush_logs():
#     try:
#         sys.stdout.flush()
#         sys.stderr.flush()
#         for h in logger.handlers:
#             try:
#                 h.flush()
#             except Exception:
#                 pass
#     except Exception:
#         pass

# # ---------------------------------------------------------------------
# # Pydantic models
# # ---------------------------------------------------------------------
# class Attachment(BaseModel):
#     name: str
#     url: str  # data URI or external URL

# class TaskRequest(BaseModel):
#     task: str
#     email: str
#     round: int
#     brief: str
#     evaluation_url: str
#     nonce: str
#     secret: str
#     attachments: List[Attachment] = []

# # ---------------------------------------------------------------------
# # FastAPI app and globals
# # ---------------------------------------------------------------------
# app = FastAPI(
#     title="Automated Task Receiver & Processor (OpenRouter)",
#     description="Receive tasks, generate code via OpenRouter LLM, deploy to GitHub Pages, and notify evaluator."
# )

# # background task management
# background_tasks_list: List[asyncio.Task] = []
# task_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
# last_received_task: Optional[dict] = None

# # OpenAI/OpenRouter constants
# OPENAI_MODEL = "openai/gpt-4.1-nano"
# OPENAI_BASE_URL = "https://aipipe.org/openrouter/v1"
# OPENAI_API_KEY = settings.OPENAI_API_KEY

# # Tool schema: use tool-calling to require structured JSON
# GENERATED_CODE_TOOL = {
#     "type": "function",
#     "function": {
#         "name": "generate_files",
#         "description": "Output the project files as JSON with keys index.html, README.md, LICENSE",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "index.html": {"type": "string", "description": "Single file HTML (Tailwind CDN)"},
#                 "README.md": {"type": "string", "description": "README content"},
#                 "LICENSE": {"type": "string", "description": "MIT license content"}
#             },
#             "required": ["index.html", "README.md", "LICENSE"],
#             "additionalProperties": False
#         }
#     }
# }

# # ---------------------------------------------------------------------
# # Helpers: secret verify
# # ---------------------------------------------------------------------
# def verify_secret(secret_from_request: str) -> bool:
#     return secret_from_request == settings.STUDENT_SECRET

# # ---------------------------------------------------------------------
# # Helpers: attachments processing
# # ---------------------------------------------------------------------
# async def process_attachment_for_llm(attachment_url: str) -> Optional[dict]:
#     """
#     Convert an attachment into a part the LLM can consume.
#     - For images: return {"inlineData": {"data": base64, "mimeType": mime}}
#     - For CSV/JSON/TXT: return {"type":"text","text": "...content..."}
#     - Returns None if unsupported or fetch failed.
#     """
#     if not attachment_url or not attachment_url.startswith(("data:", "http")):
#         logger.warning(f"Invalid attachment URL provided: {attachment_url[:50]}...")
#         return None

#     try:
#         if attachment_url.startswith("data:"):
#             match = re.search(r"data:(?P<mime>[^;]+);base64,(?P<data>.*)", attachment_url, re.IGNORECASE)
#             if not match:
#                 logger.warning("Data URI could not be parsed.")
#                 return None
#             mime = match.group("mime")
#             b64 = match.group("data")
#             if mime.startswith("image/"):
#                 return {"inlineData": {"data": b64, "mimeType": mime}}
#             else:
#                 # decode up to safe length
#                 decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
#                 if len(decoded) > 20000:
#                     decoded = decoded[:20000] + "\n\n...TRUNCATED..."
#                 return {"type": "text", "text": f"***ATTACHMENT***\nURL: {attachment_url}\nMIME: {mime}\nCONTENT:\n{decoded}"}
#         else:
#             async with httpx.AsyncClient(timeout=20) as client:
#                 resp = await client.get(attachment_url)
#                 resp.raise_for_status()
#                 mime = resp.headers.get("Content-Type", "application/octet-stream")
#                 content_bytes = resp.content
#                 b64 = base64.b64encode(content_bytes).decode("utf-8")
#                 if mime and mime.startswith("image/"):
#                     return {"inlineData": {"data": b64, "mimeType": mime}}
#                 # handle CSV/JSON/TXT heuristics
#                 lower_url = attachment_url.lower()
#                 if mime in ("text/csv", "application/json", "text/plain") or lower_url.endswith((".csv", ".json", ".txt")):
#                     decoded = content_bytes.decode("utf-8", errors="ignore")
#                     if len(decoded) > 20000:
#                         decoded = decoded[:20000] + "\n\n...TRUNCATED..."
#                     return {"type": "text", "text": f"***ATTACHMENT***\nURL: {attachment_url}\nMIME: {mime}\nCONTENT:\n{decoded}"}
#                 # unsupported: return None
#                 logger.info(f"Skipping unsupported MIME type for LLM: {mime}")
#                 return None
#     except Exception as e:
#         logger.exception(f"Failed to process attachment {attachment_url}: {e}")
#         return None

# # ---------------------------------------------------------------------
# # File helpers: save generated files & attachments locally
# # ---------------------------------------------------------------------
# def safe_makedirs(path: str):
#     os.makedirs(path, exist_ok=True)

# async def save_generated_files_locally(task_id: str, files: dict) -> str:
#     """
#     Save generated files into generated_tasks/<task_id> and return path.
#     """
#     base_dir = os.path.join(os.getcwd(), "generated_tasks")
#     task_dir = os.path.join(base_dir, task_id)
#     safe_makedirs(task_dir)
#     logger.info(f"[LOCAL_SAVE] Saving generated files to: {task_dir}")
#     for filename, content in files.items():
#         file_path = os.path.join(task_dir, filename)
#         try:
#             # write file in thread to avoid blocking event loop
#             await asyncio.to_thread(lambda p, c: open(p, "w", encoding="utf-8").write(c), file_path, content)
#             logger.info(f"   -> Saved: {filename} (bytes: {len(content)})")
#         except Exception as e:
#             logger.exception(f"Failed to save generated file {filename}: {e}")
#             raise
#     flush_logs()
#     return task_dir

# async def save_attachments_locally(task_dir: str, attachments: List[Attachment]) -> List[str]:
#     """
#     Save provided attachments into the task_dir and return list of saved filenames.
#     """
#     saved_files: List[str] = []
#     logger.info(f"[ATTACHMENTS] Processing {len(attachments)} attachments for {task_dir}")
#     async with httpx.AsyncClient(timeout=30) as client:
#         for attachment in attachments:
#             filename = attachment.name
#             url = attachment.url
#             if not filename or not url:
#                 logger.warning(f"Skipping invalid attachment: {filename}")
#                 continue
#             try:
#                 if url.startswith("data:"):
#                     match = re.search(r"base64,(.*)", url, re.IGNORECASE)
#                     if match:
#                         b64 = match.group(1)
#                         data = base64.b64decode(b64)
#                     else:
#                         logger.warning(f"Could not decode data URI for {filename}")
#                         continue
#                 else:
#                     resp = await client.get(url)
#                     resp.raise_for_status()
#                     data = resp.content
#                 file_path = os.path.join(task_dir, filename)
#                 await asyncio.to_thread(lambda p, d: open(p, "wb").write(d), file_path, data)
#                 logger.info(f"   -> Saved Attachment: {filename} (bytes: {len(data)})")
#                 saved_files.append(filename)
#             except Exception as e:
#                 logger.exception(f"Failed to save attachment {filename}: {e}")
#     flush_logs()
#     return saved_files

# # ---------------------------------------------------------------------
# # Robust removal for local paths (Windows file-locks handling)
# # ---------------------------------------------------------------------
# def remove_local_path(path: str):
#     if not os.path.exists(path):
#         return
#     logger.info(f"[CLEANUP] Removing local directory: {path}")

#     def _try_rmtree(p):
#         try:
#             shutil.rmtree(p)
#             return True
#         except FileNotFoundError:
#             return True
#         except PermissionError as e:
#             logger.warning(f"[CLEANUP] PermissionError rmtree: {e}")
#             return False
#         except Exception as e:
#             logger.exception(f"[CLEANUP] Unexpected rmtree exception: {e}")
#             return False

#     # Try multiple times; on Windows try to terminate processes with open files under path
#     for attempt in range(6):
#         ok = _try_rmtree(path)
#         if ok:
#             logger.info(f"[CLEANUP] Removed {path} on attempt {attempt+1}")
#             flush_logs()
#             return True
#         # attempt to discover processes holding file handles
#         try:
#             for proc in psutil.process_iter(['pid', 'name']):
#                 try:
#                     for f in proc.open_files():
#                         try:
#                             if os.path.commonpath([os.path.abspath(path), os.path.abspath(f.path)]) == os.path.abspath(path):
#                                 logger.warning(f"[CLEANUP] Process {proc.pid} ({proc.name()}) has open file {f.path}; attempting terminate")
#                                 try:
#                                     proc.terminate()
#                                 except Exception:
#                                     pass
#                         except Exception:
#                             continue
#                 except (psutil.AccessDenied, psutil.NoSuchProcess):
#                     continue
#         except Exception:
#             pass
#         time.sleep(1.0)
#     logger.error(f"[CLEANUP] Failed to remove {path} after retries.")
#     return False

# # ---------------------------------------------------------------------
# # GitHub helpers (create repo on round 1 or clone on subsequent rounds)
# # ---------------------------------------------------------------------
# async def setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int) -> git.Repo:
#     """
#     If round_index == 1, attempt to create the repo via GitHub API (auto_init True).
#     If repo exists or round_index > 1, clone from repo_url_auth.
#     Returns a git.Repo object.
#     """
#     github_token = settings.GITHUB_TOKEN
#     github_username = settings.GITHUB_USERNAME
#     headers = {
#         "Authorization": f"token {github_token}",
#         "Accept": "application/vnd.github.v3+json",
#         "X-GitHub-Api-Version": "2022-11-28"
#     }

#     should_clone = (round_index > 1)
#     creation_succeeded = False

#     if round_index == 1:
#         async with httpx.AsyncClient(timeout=60) as client:
#             try:
#                 logger.info(f"R1: Attempting to create remote repository '{repo_name}'...")
#                 payload = {"name": repo_name, "private": False, "auto_init": True}
#                 response = await client.post(f"{settings.GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
#                 if response.status_code == 201:
#                     creation_succeeded = True
#                 elif response.status_code == 422:
#                     # repo exists
#                     error_message = response.json().get("message", "")
#                     if "already exists on this account" in error_message:
#                         logger.warning(f"R1: Repository '{repo_name}' already exists. Switching to clone operation.")
#                         should_clone = True
#                     else:
#                         response.raise_for_status()
#                 else:
#                     response.raise_for_status()
#             except httpx.HTTPStatusError as e:
#                 logger.exception(f"GitHub API call failed during repository setup: {e.response.status_code} {e.response.text}")
#                 raise
#             except Exception as e:
#                 logger.exception(f"Unexpected error during R1 repository creation: {e}")
#                 raise

#     if should_clone or (round_index > 1 and not creation_succeeded):
#         logger.info(f"R{round_index}: Cloning existing repository from {repo_url_http}")
#         try:
#             repo = await asyncio.to_thread(git.Repo.clone_from, repo_url_auth, local_path)
#             logger.info(f"R{round_index}: Repository cloned.")
#         except git.GitCommandError as e:
#             logger.error(f"Git command failed during clone. Error: {e}")
#             raise
#     elif creation_succeeded and round_index == 1:
#         repo = git.Repo.init(local_path)
#         repo.create_remote('origin', repo_url_auth)
#         logger.info("R1: Local git repository initialized for new remote repo.")
#     else:
#         logger.error("Failed to create or clone repository due to unhandled state.")
#         raise Exception("Repository setup failed.")

#     flush_logs()
#     return repo

# async def commit_and_publish(repo: git.Repo, task_id: str, round_index: int, repo_name: str) -> dict:
#     github_username = settings.GITHUB_USERNAME
#     github_token = settings.GITHUB_TOKEN
#     headers = {
#         "Authorization": f"token {github_token}",
#         "Accept": "application/vnd.github.v3+json",
#         "X-GitHub-Api-Version": "2022-11-28"
#     }
#     repo_url_http = f"https://github.com/{github_username}/{repo_name}"

#     async with httpx.AsyncClient(timeout=60) as client:
#         try:
#             # Add & commit
#             await asyncio.to_thread(repo.git.add, A=True)
#             commit_message = f"Task {task_id} - Round {round_index}: automated update"
#             commit = await asyncio.to_thread(lambda msg: repo.index.commit(msg), commit_message)
#             commit_sha = getattr(commit, "hexsha", None) or (repo.head.commit.hexsha if repo.head and repo.head.commit else "")
#             logger.info(f"Committed changes, SHA: {commit_sha}")

#             # Ensure branch main
#             await asyncio.to_thread(lambda *args: repo.git.branch(*args), '-M', 'main')

#             def push_repo(r):
#                 r.git.push('--set-upstream', 'origin', 'main', force=True)

#             await asyncio.to_thread(push_repo, repo)
#             logger.info("Pushed changes to origin/main")

#             # Configure Pages with retries (timing issues can occur)
#             await asyncio.sleep(2)
#             pages_api_url = f"{settings.GITHUB_API_BASE}/repos/{github_username}/{repo_name}/pages"
#             pages_payload = {"source": {"branch": "main", "path": "/"}}
#             pages_max_retries = 5
#             pages_base_delay = 3

#             for attempt in range(pages_max_retries):
#                 try:
#                     pages_response = await client.get(pages_api_url, headers=headers)
#                     is_configured = (pages_response.status_code == 200)
#                     if is_configured:
#                         logger.info(f"Pages exists. Updating configuration (attempt {attempt+1})")
#                         await client.put(pages_api_url, json=pages_payload, headers=headers)
#                     else:
#                         logger.info(f"Creating Pages config (attempt {attempt+1})")
#                         await client.post(pages_api_url, json=pages_payload, headers=headers)
#                     logger.info("Pages configuration succeeded.")
#                     break
#                 except httpx.HTTPStatusError as e:
#                     text = e.response.text.lower() if e.response and e.response.text else ""
#                     if e.response.status_code == 422 and "main branch must exist" in text and attempt < pages_max_retries - 1:
#                         delay = pages_base_delay * (2 ** attempt)
#                         logger.warning(f"Timing issue configuring pages, retrying in {delay}s")
#                         await asyncio.sleep(delay)
#                         continue
#                     logger.exception(f"Failed to configure GitHub Pages: {e.response.status_code} {e.response.text}")
#                     raise

#             await asyncio.sleep(5)
#             pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/"
#             flush_logs()
#             return {"repo_url": repo_url_http, "commit_sha": commit_sha, "pages_url": pages_url}
#         except git.GitCommandError as e:
#             logger.exception("Git operation failed during deployment.")
#             raise
#         except httpx.HTTPStatusError as e:
#             logger.exception("GitHub API error during deployment.")
#             raise

# # ---------------------------------------------------------------------
# # LLM wrapper (OpenRouter) with strict system prompt and tool calling
# # ---------------------------------------------------------------------
# async def call_llm_for_code(prompt: str, task_id: str, attachment_parts: List[dict]) -> dict:
#     """
#     Use OpenAI OpenRouter client to call model with function tool schema.
#     Expects function call 'generate_files' with JSON containing index.html, README.md, LICENSE.
#     """
#     logger.info(f"[LLM_CALL] Generating code for task {task_id} using model: {OPENAI_MODEL}")

#     system_prompt = (
#         "You are an expert full-stack web engineer responsible for producing production-ready web apps "
#         "for GitHub Pages deployment. You MUST ALWAYS return a single function call to 'generate_files' "
#         "containing exactly three keys: 'index.html', 'README.md', and 'LICENSE'. No text outside that JSON is allowed.\n\n"
#         "ROUND LOGIC:\n"
#         "- Round 1: Create a complete, single-file responsive Tailwind web app (index.html) plus README.md and MIT LICENSE.\n"
#         "- Round 2+: Make minimal precise edits to the previous project: preserve structure, styling, and behavior; update only what's required by the brief.\n\n"
#         "RULES:\n"
#         "1) index.html must be a single HTML file using Tailwind CDN and vanilla JS only.\n"
#         "2) If CSV/JSON/TXT attachments are present, embed them inline (script type='application/json' or type='text/csv') and visualize them (table or chart).\n"
#         "3) README.md must explain purpose, data usage, and include Live Demo link.\n"
#         "4) LICENSE must be MIT with placeholders [year] and [author].\n"
#         "5) Never reference external private APIs or require runtime keys. App must work client-side and offline after deployment.\n"
#         "6) Output exactly one JSON argument object to 'generate_files' containing complete file contents (index.html, README.md, LICENSE).\n"
#     )

#     # Build content for user message; include attachments as text/image parts
#     user_content_parts: List[Dict[str, Any]] = []
#     user_content_parts.append({"type": "text", "text": prompt})
#     for part in attachment_parts:
#         if part.get("inlineData"):
#             mime = part["inlineData"].get("mimeType", "application/octet-stream")
#             data_b64 = part["inlineData"].get("data", "")
#             user_content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data_b64}"}})
#         elif part.get("type") == "text" and part.get("text"):
#             user_content_parts.append({"type": "text", "text": part["text"]})

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_content_parts}
#     ]

#     max_retries = 3
#     base_delay = 1

#     if not OPENAI_API_KEY:
#         raise Exception("OPENAI_API_KEY not configured in environment.")

#     try:
#         client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, http_client=httpx.Client(timeout=90.0))
#     except Exception as e:
#         logger.error(f"Failed to initialize OpenRouter client: {e}")
#         raise

#     for attempt in range(max_retries):
#         try:
#             response = client.chat.completions.create(
#                 model=OPENAI_MODEL,
#                 messages=messages,
#                 tools=[GENERATED_CODE_TOOL],
#                 tool_choice={"type": "function", "function": {"name": "generate_files"}},
#                 temperature=0.0
#             )
#             # Response parsing depends on OpenAI client wrapper structure
#             response_message = response.choices[0].message
#             tool_calls = getattr(response_message, "tool_calls", None) or []
#             if tool_calls and tool_calls[0].function.name == "generate_files":
#                 json_text = tool_calls[0].function.arguments
#                 generated_files = json.loads(json_text)
#                 # Validate expected keys
#                 for key in ("index.html", "README.md", "LICENSE"):
#                     if key not in generated_files:
#                         raise ValueError(f"Missing key in generated_files: {key}")
#                 logger.info(f"[LLM_CALL] Successfully generated files on attempt {attempt+1}")
#                 flush_logs()
#                 return generated_files
#             else:
#                 raise ValueError("Model did not return a 'generate_files' function call.")
#         except Exception as e:
#             logger.warning(f"[LLM_CALL] Attempt {attempt+1} error: {e}")
#             if attempt < max_retries - 1:
#                 await asyncio.sleep(base_delay * (2 ** attempt))
#     logger.error("[LLM_CALL] Exhausted retries without successful generation.")
#     raise Exception("LLM generation failed after retries")

# # ---------------------------------------------------------------------
# # Notifier to evaluation server
# # ---------------------------------------------------------------------
# async def notify_evaluation_server(evaluation_url: str, email: str, task_id: str, round_index: int, nonce: str, repo_url: str, commit_sha: str, pages_url: str) -> bool:
#     # Skip invalid/test endpoints to avoid 403s during local testing
#     if not evaluation_url or "example.com" in evaluation_url or evaluation_url.strip() == "":
#         logger.warning("[NOTIFY] Skipping notification due to invalid or example URL.")
#         return False

#     payload = {
#         "email": email,
#         "task": task_id,
#         "round": round_index,
#         "nonce": nonce,
#         "repo_url": repo_url,
#         "commit_sha": commit_sha,
#         "pages_url": pages_url
#     }

#     max_retries = 3
#     base_delay = 1
#     logger.info(f"[NOTIFY] Notifying evaluation server at {evaluation_url}")
#     for attempt in range(max_retries):
#         try:
#             async with httpx.AsyncClient(timeout=15) as client:
#                 resp = await client.post(evaluation_url, json=payload)
#                 resp.raise_for_status()
#                 logger.info(f"[NOTIFY] Notification succeeded: {resp.status_code}")
#                 flush_logs()
#                 return True
#         except httpx.HTTPStatusError as e:
#             logger.warning(f"[NOTIFY] HTTP error attempt {attempt+1}: {e}")
#         except httpx.RequestError as e:
#             logger.warning(f"[NOTIFY] Request error attempt {attempt+1}: {e}")
#         if attempt < max_retries - 1:
#             delay = base_delay * (2 ** attempt)
#             logger.info(f"[NOTIFY] Retrying in {delay}s")
#             await asyncio.sleep(delay)
#     logger.error("[NOTIFY] Failed to notify evaluation server after retries.")
#     flush_logs()
#     return False

# # ---------------------------------------------------------------------
# # Main orchestration function (per-task)
# # ---------------------------------------------------------------------
# async def generate_files_and_deploy(task_data: TaskRequest):
#     acquired = False
#     try:
#         await task_semaphore.acquire()
#         acquired = True
#         logger.info(f"[PROCESS START] Task: {task_data.task} Round: {task_data.round}")
#         flush_logs()

#         task_id = task_data.task
#         email = task_data.email
#         round_index = task_data.round
#         brief = task_data.brief
#         evaluation_url = task_data.evaluation_url
#         nonce = task_data.nonce
#         attachments = task_data.attachments or []

#         repo_name = task_id.replace(" ", "-").lower()
#         github_username = settings.GITHUB_USERNAME
#         github_token = settings.GITHUB_TOKEN
#         repo_url_auth = f"https://{github_username}:{github_token}@github.com/{github_username}/{repo_name}.git"
#         repo_url_http = f"https://github.com/{github_username}/{repo_name}"

#         base_dir = os.path.join(os.getcwd(), "generated_tasks")
#         local_path = os.path.join(base_dir, task_id)

#         # Cleanup local_path if present to ensure empty dir
#         if os.path.exists(local_path):
#             try:
#                 await asyncio.to_thread(remove_local_path, local_path)
#             except Exception as e:
#                 logger.exception(f"Cleanup failed for {local_path}: {e}")
#                 # Don't proceed if cleanup fails significantly
#                 raise

#         safe_makedirs(local_path)

#         # Setup repo (clone or init)
#         repo = await setup_local_repo(
#             local_path=local_path,
#             repo_name=repo_name,
#             repo_url_auth=repo_url_auth,
#             repo_url_http=repo_url_http,
#             round_index=round_index
#         )

#         # Prepare attachment parts for LLM
#         attachment_parts: List[dict] = []
#         attachment_list_for_prompt: List[str] = []
#         for attachment in attachments:
#             part = await process_attachment_for_llm(attachment.url)
#             if part:
#                 attachment_parts.append(part)
#             attachment_list_for_prompt.append(attachment.name)
#         logger.info(f"[LLM_INPUT] Attachment parts: {len(attachment_parts)} Attachments names: {attachment_list_for_prompt}")

#         # Build LLM prompt
#         if round_index > 1:
#             llm_prompt = (
#                 f"UPDATE INSTRUCTION (ROUND {round_index}): Modify the existing project files "
#                 f"(index.html, README.md, LICENSE) to satisfy this brief: '{brief}'. "
#                 "Perform minimal, precise edits only; keep layout, styling, and behavior intact. "
#                 "Return the full content for index.html, README.md, and LICENSE."
#             )
#         else:
#             llm_prompt = (
#                 f"CREATE INSTRUCTION (ROUND {round_index}): Build a complete, single-file responsive Tailwind web app for: '{brief}'. "
#                 "Provide index.html (single file using Tailwind CDN), README.md, and MIT LICENSE. If attachments are included, incorporate them into the app (tables/visualizations/images)."
#             )

#         if attachment_list_for_prompt:
#             llm_prompt += f" Additional project files provided: {', '.join(attachment_list_for_prompt)}"

#         # Call LLM
#         generated_files = await call_llm_for_code(llm_prompt, task_id, attachment_parts)

#         # Save generated files locally (overwrite)
#         task_dir = await save_generated_files_locally(task_id, generated_files)

#         # Save attachments into repository
#         await save_attachments_locally(task_dir, attachments)

#         # Commit and publish
#         deployment_info = await commit_and_publish(
#             repo=repo,
#             task_id=task_id,
#             round_index=round_index,
#             repo_name=repo_name
#         )

#         repo_url = deployment_info["repo_url"]
#         commit_sha = deployment_info["commit_sha"]
#         pages_url = deployment_info["pages_url"]

#         logger.info(f"[DEPLOYMENT] Success. Repo: {repo_url} Pages: {pages_url}")

#         # Notify evaluation server (skip example.com)
#         await notify_evaluation_server(
#             evaluation_url=evaluation_url,
#             email=email,
#             task_id=task_id,
#             round_index=round_index,
#             nonce=nonce,
#             repo_url=repo_url,
#             commit_sha=commit_sha,
#             pages_url=pages_url
#         )

#     except Exception as exc:
#         logger.exception(f"[CRITICAL FAILURE] Task {task_data.task} failed: {exc}")
#     finally:
#         if acquired:
#             task_semaphore.release()
#         flush_logs()
#         logger.info(f"[PROCESS END] Task: {task_data.task} Round: {task_data.round}")

# # ---------------------------------------------------------------------
# # Background callback
# # ---------------------------------------------------------------------
# def _task_done_callback(task: asyncio.Task):
#     try:
#         exc = task.exception()
#         if exc:
#             logger.error(f"[BACKGROUND TASK] Task finished with exception: {exc}")
#         else:
#             logger.info("[BACKGROUND TASK] Task finished successfully.")
#     except asyncio.CancelledError:
#         logger.warning("[BACKGROUND TASK] Task was cancelled.")
#     finally:
#         flush_logs()

# # ---------------------------------------------------------------------
# # FastAPI endpoints
# # ---------------------------------------------------------------------
# @app.post("/ready", status_code=200)
# async def receive_task(task_data: TaskRequest, background_tasks: BackgroundTasks, request: Request):
#     global last_received_task, background_tasks_list

#     if not verify_secret(task_data.secret):
#         logger.warning(f"Unauthorized attempt for task {task_data.task} from {request.client.host if request.client else 'unknown'}")
#         raise HTTPException(status_code=401, detail="Unauthorized: Secret mismatch")

#     last_received_task = {
#         "task": task_data.task,
#         "email": task_data.email,
#         "round": task_data.round,
#         "brief": (task_data.brief[:250] + "...") if len(task_data.brief) > 250 else task_data.brief,
#         "time": datetime.utcnow().isoformat() + "Z"
#     }

#     bg_task = asyncio.create_task(generate_files_and_deploy(task_data))
#     bg_task.add_done_callback(_task_done_callback)
#     background_tasks_list.append(bg_task)

#     # Include a no-op background task so FastAPI knows we scheduled work
#     background_tasks.add_task(lambda: None)

#     logger.info(f"Received task {task_data.task}. Background processing scheduled.")
#     flush_logs()

#     return JSONResponse(status_code=200, content={"status": "processing_scheduled", "message": f"Task {task_data.task} received and background processing started."})

# @app.get("/")
# async def root():
#     return {"message": "Task Receiver Service running. POST /ready to submit."}

# @app.get("/status")
# async def get_status():
#     if last_received_task:
#         return {"last_received_task": last_received_task, "running_background_tasks": len([t for t in background_tasks_list if not t.done()])}
#     return {"message": "Awaiting first task submission to /ready"}

# @app.get("/health")
# async def health():
#     return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

# @app.get("/logs")
# async def get_logs(lines: int = Query(200, ge=1, le=5000)):
#     path = settings.LOG_FILE_PATH
#     if not os.path.exists(path):
#         return PlainTextResponse("Log file not found.", status_code=404)
#     try:
#         with open(path, "rb") as f:
#             f.seek(0, os.SEEK_END)
#             file_size = f.tell()
#             buffer = bytearray()
#             block_size = 1024
#             blocks = 0
#             while file_size > 0 and len(buffer) < lines * 2000 and blocks < 1024:
#                 read_size = min(block_size, file_size)
#                 f.seek(file_size - read_size)
#                 buffer.extend(f.read(read_size))
#                 file_size -= read_size
#                 blocks += 1
#             text = buffer.decode(errors="ignore").splitlines()
#             last_lines = "\n".join(text[-lines:])
#             return PlainTextResponse(last_lines)
#     except Exception as e:
#         logger.exception(f"Error reading log file: {e}")
#         return PlainTextResponse(f"Error reading log file: {e}", status_code=500)

# # ---------------------------------------------------------------------
# # Keep-alive loop (for long-running spaces)
# # ---------------------------------------------------------------------
# @app.on_event("startup")
# async def startup_event():
#     async def keep_alive():
#         while True:
#             try:
#                 logger.info("[KEEPALIVE] Service heartbeat")
#                 flush_logs()
#             except Exception:
#                 pass
#             await asyncio.sleep(settings.KEEP_ALIVE_INTERVAL_SECONDS)
#     asyncio.create_task(keep_alive())

# # ---------------------------------------------------------------------
# # Graceful shutdown
# # ---------------------------------------------------------------------
# @app.on_event("shutdown")
# async def shutdown_event():
#     logger.info("[SHUTDOWN] Waiting for background tasks to finish (graceful shutdown)...")
#     for t in background_tasks_list:
#         if not t.done():
#             try:
#                 t.cancel()
#             except Exception:
#                 pass
#     await asyncio.sleep(0.5)
#     flush_logs()

# # ---------------------------------------------------------------------
# # End of file
# # ---------------------------------------------------------------------
