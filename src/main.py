import sys
import os
import numpy as np
import time
import asyncio
import json
from typing import AsyncGenerator

import argparse
import requests
import fastapi
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
import asyncio
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import time, datetime
import os
import random
import base64
import aiohttp
import re

from run_infer import run_stream

app = FastAPI()

@app.post("/v1/llm_data_analyze_stream")
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    print(f'receive request time: {time.strftime("%Y-%m-%d %H:%I:%S", time.localtime( time.time() ) )}')
    print(f"request_dict: {request_dict!r}")
    
    recordId = request_dict['recordId']
    modelId = request_dict['modelId']
    stream = request_dict['stream']
    filePath = request_dict['filePath']
    history = request_dict['history']
    params = request_dict['params']
    userId = request_dict['userId']
    prompt = request_dict['prompt']

    file_paths = filePath
    uid = userId
    session_id = recordId

    for file_path in file_paths:
        if file_path == '':
            file_paths = []
        break

    try:
        # Abort the request if the client disconnects.
        background_tasks = BackgroundTasks()
        return EventSourceResponse(main_stream(uid, session_id, file_paths, prompt, history, request))
    except Exception as e:
        print('generate error: {}'.format(e))


async def main_stream(uid, session_id, file_paths, prompt, history, request):
    async for resJson in run_stream(uid, session_id, file_paths, prompt, history, request):
        yield ServerSentEvent(json.dumps(resJson, ensure_ascii=False), event='delta')
        await asyncio.sleep(0.0)

def deploy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8006)
    
    args = parser.parse_args()
    
    TIMEOUT_KEEP_ALIVE = 5 # seconds.
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE
                )

if __name__ == "__main__":
    deploy()
