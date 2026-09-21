from pathlib import Path
import json,time,traceback,hashlib
import torch
from PIL import Image
from transformers import AutoModel,AutoTokenizer
r=Path('/root/tcei_competition_20260919');events=[json.loads(x) for x in (r/'t1_initial03_01/events.jsonl').read_text().splitlines() if x.strip()];event=next(e for e in events if e['status']=='infer_started')
image_path=Path('/root/gpufree-data/tcei_competition_20260919/no_action03_01/events')/event['image'];im=Image.open(image_path).convert('RGB')
out={'recorded_input_only':True,'robot_commands_sent':False,'original_request_id':event['request_id'],'instruction':event['instruction'],'input_image':str(image_path),'input_png_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),'image_size':list(im.size),'prompt':event['prompt'],'memory_events':[]}
def memory(label,**fields):
 free,total=torch.cuda.mem_get_info();row={'phase':label,'wall_time':time.time(),'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'free_device':free,**fields};out['memory_events'].append(row);(r/'nine_vbatch1_beam1_progress.txt').write_text(json.dumps(row,indent=2));print('MEMORY',json.dumps(row),flush=True)
try:
 model=AutoModel.from_pretrained('/root/inference/FM9G4B-V',trust_remote_code=True,attn_implementation='sdpa',torch_dtype=torch.bfloat16).eval().to('cuda');tokenizer=AutoTokenizer.from_pretrained('/root/inference/FM9G4B-V',trust_remote_code=True)
 out['original_vision_batch_size']=model.config.vision_batch_size;model.config.vision_batch_size=1;out['vision_batch_size']=1;out['num_beams']=1;out['generation_do_sample']=model.llm.generation_config.do_sample;out['max_slice_nums']=model.config.slice_config.max_slice_nums;out['vision_attention']=model.config.vision_config._attn_implementation;out['language_attention']=model.llm.config._attn_implementation
 def before(module,args):memory('vision_before',input_shape=list(args[0].shape))
 def after(module,args,result):memory('vision_after',output_shape=list(result.last_hidden_state.shape))
 model.vpm.register_forward_pre_hook(before);model.vpm.register_forward_hook(after)
 torch.cuda.reset_peak_memory_stats();memory('loaded');started=time.monotonic()
 with torch.inference_mode():answer=model.chat(image=None,msgs=[{'role':'user','content':[im,event['prompt']]}],tokenizer=tokenizer,max_new_tokens=400,sampling=False,num_beams=1)
 torch.cuda.synchronize();out.update(status='model_answer',answer=str(answer),elapsed_seconds=time.monotonic()-started);memory('completed')
except Exception as error:
 out.update(status='runtime_failure',error=repr(error),traceback=traceback.format_exc());memory('failure')
finally:
 (r/'nine_vbatch1_beam1_result.txt').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('PROBE_COMPLETE',out['status'],flush=True)
