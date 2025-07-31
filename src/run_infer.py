import json
import os
import argparse

from table.POT_reason import POT_Reasoner
from sample_format.sample import Sample
import time
import asyncio
from uuid import uuid4
import warnings
warnings.filterwarnings("ignore")

class RunConfig:
    def __init__(self,
        model_url='http://127.0.0.1:8027/v1/completions',
        model_name='JT-DA-8B',
        model_path='./JT-DA-8B',
        cwd='./output/',
        uid='test',
        session_id='345',
        req=None,
        if_stream=1,
        few_csv=4,
        cell_max_len=50,
        deploy_location='CESHI'
    ):
        self.model_url = model_url
        self.model_name = model_name
        self.model_path = model_path
        self.cwd = cwd
        self.uid = uid
        self.session_id = session_id
        self.req = req
        self.if_stream = if_stream
        self.few_csv = few_csv
        self.cell_max_len=cell_max_len
        self.deploy_location=deploy_location
    
    
async def infer_stream(config, sample, session_id):
    reasoner = POT_Reasoner(config, sample)
    async for resJson in reasoner.reason_stream(sample, think=True):
        yield resJson
    
# prepare config
async def run_stream(uid, session_id, file_paths, prompt, history, request):
    config = RunConfig(uid=uid, session_id=session_id, req=request)
    sample = Sample(file_paths=file_paths, input_prompt=prompt, history=history, config=config)
    
    async for resJson in infer_stream(config, sample, session_id):
        yield resJson
    print('output complete', flush=True)
