from pathlib import Path
import json,time,traceback
from PIL import Image
from transformers import AutoProcessor
r=Path('/root/tcei_competition_20260919'); run=Path('/root/gpufree-data/tcei_competition_20260919/no_action04_01')
rows=[json.loads(x) for x in (r/'t1_initial04_01/events.jsonl').read_text().splitlines()]
p=AutoProcessor.from_pretrained('/root/inference/FM9G4B-V',trust_remote_code=True)
out={'robot_commands_sent':False,'model_loaded':False,'max_input_tokens':8192,'requests':[]}
for e in [x for x in rows if x.get('status')=='infer_started']:
 related=[x for x in rows if x.get('request_id')==e['request_id']]; answer=next(x['answer'] for x in related if x.get('status')=='model_answer'); feedback=next(x['feedback'] for x in related if x.get('status')=='repair_started')
 repair=e['prompt']+chr(10)+'上一次输出未通过校验：'+answer+chr(10)+'校验原因：'+feedback+chr(10)+'请重新检查用户任务：'+e['instruction']+'。核对所有限定词、空间条件、类别、目标侧、数量和ID；不得忽略限定词。重新输出11键JSON；无法确定时输出needs_confirmation及原因。'
 image=Image.open(run/e['image']).convert('RGB'); result={'id':e['request_id'],'attempts':[]}; seq=[]
 for label,prompt in [('first',e['prompt']),('repair',repair)]:
  chat=p.tokenizer.apply_chat_template([{'role':'user','content':'(<image>./</image>)'+chr(10)+prompt}],tokenize=False,add_generation_prompt=True)
  full=p([chat],[[image]],return_tensors='pt',max_length=None)['input_ids'][0]; limited=p([chat],[[image]],return_tensors='pt',max_length=8192)['input_ids'][0]; decoded=p.tokenizer.decode(limited.tolist()); seq.append(limited.tolist()); result['attempts'].append({'label':label,'full_tokens':len(full),'effective_tokens':len(limited),'truncated':len(full)!=len(limited),'task_present':e['instruction'] in decoded,'tail':decoded[-350:],'feedback_present':feedback in decoded})
 result['effective_inputs_identical']=seq[0]==seq[1]; out['requests'].append(result)
(r/'nine_tokenization04.txt').write_text(json.dumps(out,ensure_ascii=False,indent=2)); print('TOKENIZATION_DONE',flush=True)
