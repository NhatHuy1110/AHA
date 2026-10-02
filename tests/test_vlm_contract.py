"""Processor-independent VLM contracts. Does NOT assert actual Qwen forward works."""
import numpy as np
import pytest
import torch
from aha.vlm import sampled_seconds,encode,parse

def test_vlm_samples_are_label_free_unique_bounded():
    phase=np.ones((1000,12),np.float32)*.01; phase[400:600,3]=.9
    config=dict(proposals=dict(topk=16,nms_seconds=30,coverage=8,radius=32))
    times=sampled_seconds(phase,3,config,budget=48)
    assert times==sorted(set(times)) and 0<len(times)<=48
    assert min(times)>=0 and max(times)<1000

class Batch(dict):
    def to(self,device):
        return self

class Processor:
    def __init__(self,mismatch=False):
        self.mismatch=mismatch
    def apply_chat_template(self,messages,**kwargs):
        return 'full' if messages[-1]['role']=='assistant' else 'prompt'
    def __call__(self,text,**kwargs):
        seq=[1,2,3] if text[0]=='prompt' else ([9,2,3,4,5] if self.mismatch else [1,2,3,4,5])
        return Batch(input_ids=torch.tensor([seq]))

def test_sft_supervises_only_answer_tokens():
    result=encode(Processor(),[dict(role='user')],[],device='cpu',target=dict(answer='A',second=1))
    assert result['labels'].tolist()==[[-100,-100,-100,4,5]]

def test_sft_rejects_mismatched_prefix():
    with pytest.raises(ValueError,match='prefix mismatch'):
        encode(Processor(True),[dict(role='user')],[],device='cpu',target=dict(answer='A',second=1))

def test_reply_parsing_is_lenient_but_never_invents_values():
    assert parse('{"answer": "B", "second": 120}')==('B',120)
    assert parse('```json\n{"answer":"c","second":"45"}\n```')==('C',45)
    assert parse('The answer is D. second: 300')==('D',300)
    assert parse('I cannot tell.')==(None,None)
    assert parse('{"answer": "E", "second": 1.5}')==(None,None)
