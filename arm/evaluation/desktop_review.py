#!/usr/bin/python3
"""Read-only ROS review window; human notes never select or command the robot."""
import argparse
import base64
import collections
import copy
import io
import json
import os
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import cv2
from PIL import Image,ImageDraw
import rospy
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String,Bool
from cv_bridge import CvBridge

NAMES={'Torch':'军用手电筒','Smokegrenade':'烟雾弹','Magazine':'弹夹','Grenade':'手雷','CompressedFood':'压缩干粮'}
STAGES={'observation_started':'开始让位','observation_completed':'让位完成','infer_started':'九格开始分析输入',
        'model_answer':'九格返回原始结果','validation_failed':'模型结果校验未通过','plan_published':'校验通过，计划发布',
        'plan_accepted':'控制器接受计划','grasp_evidence_preflight':'试抬证据能力检查',
        'grasp_candidate_set':'生成抓取候选点','approach':'实际接近','descend':'实际下降',
        'grasp_contact_confirmed':'接触窗口确认','grasp_verified':'试抬抓稳确认','released':'已释放',
        'grasp_preload':'渐进预紧完成','placement_pending_verification':'已释放，放置待核验',
        'task_pending_verification':'本任务已释放，继续后续任务','placement_verification_expired':'放置证据尚未补齐',
        'placement_verified':'放置和输送确认','task_succeeded':'任务完成','task_failed':'任务失败',
        'controlled_stop_verified':'已确认停止','drop_detected':'夹持丢失信号',
        'holding_feedback_lost':'持续失去夹持','drop_recovery_started':'开始夹持丢失恢复'}


def tk_photo(picture):
    data=io.BytesIO();picture.save(data,format='PNG')
    return tk.PhotoImage(data=base64.b64encode(data.getvalue()).decode('ascii'))


class Review:
    def __init__(self,config,out):
        self.config_path=Path(config);self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.bridge=CvBridge();self.frames={};self.history=collections.OrderedDict()
        self.scene={};self.stop={};self.stop_received=0.;self.capture=None;self.requests=collections.OrderedDict();self.latest_request=None
        self.events=collections.deque(maxlen=120);self.seen=set();self.preview_reasons=collections.Counter()
        self.verified_releases=set();self.pending_releases=set()
        self.grasp=None;self.grasp_image=None;self.config={};self.config_key=None;self.frozen=False;self.context={}
        self.photos={};self.pictures={};self.rendered={};self.last_model_key=None
        self.root=tk.Tk();self.root.title('TCEI 联合核对：YOLO · 九格 · Isaac')
        self.root.geometry(os.environ.get('TCEI_REVIEW_GEOMETRY','1960x1120+10+35'));self.root.minsize(1400,900)
        style=ttk.Style();style.configure('.',font=('Sans',12));style.configure('Heading.TLabel',font=('Sans',16,'bold'))
        self.header=tk.StringVar();ttk.Label(self.root,textvariable=self.header,style='Heading.TLabel',wraplength=1900).pack(fill='x',padx=12,pady=8)
        bar=ttk.Frame(self.root);bar.pack(fill='x',padx=12)
        self.freeze_button=ttk.Button(bar,text='固定当前画面以便核对',command=self.toggle);self.freeze_button.pack(side='left')
        ttk.Button(bar,text='查看上轮放置原图',command=self.show_review_image).pack(side='left',padx=8)
        ttk.Label(bar,text='只固定显示，不改变机械臂状态；双击图像可看原尺寸。').pack(side='left',padx=18)
        self.book=ttk.Notebook(self.root);self.book.pack(fill='both',expand=True,padx=10,pady=8)
        live=ttk.Frame(self.book);model=ttk.Frame(self.book);points=ttk.Frame(self.book);details=ttk.Frame(self.book)
        for page,name in [(live,'实时相机与 YOLO'),(model,'九格实际输入与结果'),(points,'抓取点与待核对项'),(details,'完整输入文本与执行记录')]:self.book.add(page,text=name)
        self.book.select(model)
        self.labels={};self.captions={}
        for key,page,column,title in [('raw',live,0,'相机原图（实时）'),('yolo',live,1,'YOLO 编号与边界（实时）'),
                                      ('model',model,0,'本次九格收到的原始编号图'),('grasp',points,0,'实际同帧的抓取候选点')]:
            box=ttk.Frame(page);box.grid(row=0,column=column,sticky='nsew',padx=8,pady=5)
            self.captions[key]=tk.StringVar(value=title)
            ttk.Label(box,textvariable=self.captions[key],wraplength=930).pack(anchor='w')
            label=ttk.Label(box,text='等待对应画面…',anchor='center');label.pack(fill='both',expand=True)
            label.bind('<Double-Button-1>',lambda event,k=key:self.full_image(k));self.labels[key]=label
            page.columnconfigure(column,weight=1);page.rowconfigure(0,weight=1)
        self.table=ttk.Treeview(live,columns=('id','name','confidence','pixel','axis'),show='headings',height=7)
        for key,title in [('id','编号'),('name','YOLO 类别'),('confidence','置信度'),('pixel','图像坐标'),('axis','物体长轴角度')]:
            self.table.heading(key,text=title);self.table.column(key,width=170,anchor='center')
        self.table.grid(row=1,column=0,columnspan=2,sticky='ew',padx=8,pady=8)
        self.answer=ScrolledText(model,width=68,height=32,font=('Sans',13),wrap='word')
        self.answer.grid(row=0,column=1,sticky='nsew',padx=8,pady=8);model.columnconfigure(1,weight=1)
        self.questions=ScrolledText(points,width=68,height=32,font=('Sans',13),wrap='word')
        self.questions.grid(row=0,column=1,sticky='nsew',padx=8,pady=8);points.columnconfigure(1,weight=1)
        self.prompt=ScrolledText(details,height=22,font=('Sans',11),wrap='word');self.prompt.pack(fill='both',expand=True)
        self.timeline=ScrolledText(details,height=13,font=('Sans',11),wrap='word');self.timeline.pack(fill='both',expand=True)
        footer=ttk.Frame(self.root);footer.pack(fill='x',padx=12,pady=8)
        ttk.Label(footer,text='你的核对意见：').pack(side='left');self.note=ttk.Entry(footer,width=88);self.note.pack(side='left',fill='x',expand=True,padx=8)
        for verdict in ('判断正确','有问题','无法确定'):
            ttk.Button(footer,text=verdict,command=lambda v=verdict:self.feedback(v)).pack(side='left',padx=4)
        self.feedback_status=tk.StringVar(value='意见仅保存为开发核对记录，不发送任何机械臂指令。')
        ttk.Label(self.root,textvariable=self.feedback_status).pack(anchor='w',padx=12,pady=3)
        rospy.init_node('tcei_review_display',anonymous=True,disable_signals=True)
        self.subs=[rospy.Subscriber(topic,RosImage,lambda m,k=key:self.on_image(k,m),queue_size=1,buff_size=2**24)
                   for key,topic in [('raw','/Jaka/camera/rgb'),('yolo','/tcei/annotated_image')]]
        self.subs.extend(rospy.Subscriber(topic,String,lambda m,t=topic:self.on_json(t,m),queue_size=100)
                         for topic in ['/tcei/candidates','/tcei/task_status','/tcei/nine_status','/tcei/stop_ack'])
        self.subs.append(rospy.Subscriber('/Jaka/gripper_is_captured',Bool,lambda m:setattr(self,'capture',bool(m.data)),queue_size=1))
        self.root.protocol('WM_DELETE_WINDOW',self.close);self.reload_config();self.tick()
        (self.out/'ready.json').write_text(json.dumps({'time':time.time(),'config':str(self.config_path),'control_publishers':0}))

    def on_image(self,key,msg):
        with self.lock:
            row=(msg,time.monotonic());self.frames[key]=row
            if key=='yolo':
                stamp=msg.header.stamp.to_nsec();self.history[stamp]=row
                while len(self.history)>32:self.history.popitem(last=False)

    def on_json(self,topic,msg):
        try:value=json.loads(msg.data)
        except (ValueError,TypeError):return
        if not isinstance(value,dict):return
        with self.lock:
            if topic=='/tcei/candidates':self.scene=value
            elif topic=='/tcei/stop_ack':self.stop=value;self.stop_received=time.monotonic()
            else:self.event(value)

    def event(self,value):
        status=value.get('status');rid=value.get('request_id')
        candidate=value.get('candidate')
        stable=value.get('stable_id') or (candidate.get('stable_id') if isinstance(candidate,dict) else None)
        release_key=(value.get('round_id'),stable)
        if stable and status=='placement_verified':
            self.verified_releases.add(release_key);self.pending_releases.discard(release_key)
        elif stable and status=='placement_pending_verification' and release_key not in self.verified_releases:
            self.pending_releases.add(release_key)
        key=(status,rid,value.get('monotonic',value.get('time')))
        if key in self.seen:return
        self.seen.add(key)
        if len(self.seen)>10000:self.seen={key}
        if status=='infer_started':
            self.requests[rid]={'input':copy.deepcopy(value)};self.latest_request=rid
            while len(self.requests)>12:self.requests.popitem(last=False)
        elif rid in self.requests and status in ('model_answer','validation_failed','plan_published','dry_run_validated','rejected'):
            self.requests[rid][status]=copy.deepcopy(value)
        if status=='grasp_candidate_set':self.grasp=copy.deepcopy(value);self.grasp_image=None
        if status=='grasp_preview':self.preview_reasons[value.get('proof',{}).get('reason','通过')]+=1
        if status in STAGES:
            self.events.append(time.strftime('%H:%M:%S',time.localtime(value.get('time',time.time())))+'  '+STAGES[status]+('：'+str(value['reason']) if value.get('reason') else ''))

    def reload_config(self):
        try:
            config=json.loads(self.config_path.read_text());key=json.dumps(config,sort_keys=True)
            if key==self.config_key:return
            allowed=Path(os.environ['TCEI_RUNS']).resolve()
            for name in ('stack_dir','episode_dir'):
                if allowed not in Path(config[name]).resolve().parents:raise ValueError('review path outside task run root')
            if config.get('campaign_file') and allowed not in Path(config['campaign_file']).resolve().parents:
                raise ValueError('campaign path outside task run root')
            self.config=config;self.config_key=key;self.stop={};self.stop_received=0.
            self.requests.clear();self.latest_request=None;self.events.clear();self.preview_reasons.clear();self.seen.clear()
            self.verified_releases.clear();self.pending_releases.clear()
            self.grasp=None;self.grasp_image=None;self.rendered.clear();self.context={};self.frames.clear();self.history.clear();self.scene={}
            for image_key in ('model','grasp'):
                self.pictures.pop(image_key,None);self.labels[image_key].configure(image='',text='等待本轮对应画面…')
            path=Path(config['episode_dir'])/'events.jsonl'
            if path.exists():
                with path.open() as stream:
                    for line in stream:
                        try:self.event(json.loads(line))
                        except (ValueError,TypeError):continue
        except Exception as error:self.feedback_status.set('核对配置读取失败：'+str(error))

    def photo(self,key,picture):
        self.pictures[key]=picture.copy();display=picture.copy();display.thumbnail((940,535),Image.LANCZOS)
        self.photos[key]=tk_photo(display);self.labels[key].configure(image=self.photos[key],text='')

    def full_image(self,key):
        if key not in self.pictures:return
        window=tk.Toplevel(self.root);caption=self.captions.get(key)
        window.title(caption.get() if caption is not None else str(key))
        picture=tk_photo(self.pictures[key]);label=ttk.Label(window,image=picture);label.image=picture;label.pack()

    @staticmethod
    def text(widget,value):
        if getattr(widget,'last_value',None)==value:return
        widget.configure(state='normal');widget.delete('1.0','end');widget.insert('end',value);widget.configure(state='disabled');widget.last_value=value

    def model_picture(self,record):
        name=record.get('input',{}).get('image')
        if not isinstance(name,str) or Path(name).name!=name:return
        path=Path(self.config['stack_dir'])/'events'/name
        if path.exists() and self.rendered.get('model')!=str(path):
            with Image.open(path) as source:self.photo('model',source.convert('RGB'))
            self.rendered['model']=str(path)

    def grasp_picture(self):
        if not self.grasp:return
        candidate=self.grasp['candidate'];stamp=candidate.get('image_stamp');picture=None
        row=self.history.get(round(stamp*1e9)) if isinstance(stamp,(int,float)) else None
        if row:
            array=self.bridge.imgmsg_to_cv2(row[0],'bgr8');picture=Image.fromarray(cv2.cvtColor(array,cv2.COLOR_BGR2RGB))
        elif self.grasp_image is None and stamp is not None:
            folder=Path(self.config['episode_dir']).parent/'rgbd';index=folder/'index.jsonl'
            if index.exists():
                for line in index.read_text().splitlines():
                    item=json.loads(line)
                    if abs(item['stamp']-stamp)<1e-6:
                        with Image.open(folder/(item['stem']+'.jpg')) as source:picture=source.convert('RGB')
                        break
            self.grasp_image=picture if picture is not None else False
        elif self.grasp_image is not False:picture=self.grasp_image
        if picture is None:
            self.captions['grasp'].set('未保存抓取候选的同帧图像；不叠加到其他时刻画面。');return
        picture=picture.copy();draw=ImageDraw.Draw(picture)
        for number,item in enumerate(self.grasp['generated']['candidates']):
            x,y=item['pose_candidate']['pixel'];draw.ellipse((x-5,y-5,x+5,y+5),outline='#ff00ff',width=2);draw.text((x+7,y-9),'C'+str(number),fill='#ff00ff')
        self.photo('grasp',picture);self.captions['grasp'].set('紫色 C0…为实际候选点；源画面仿真时刻 %.6f 秒'%stamp)

    def render(self):
        for key,title in [('raw','相机原图'),('yolo','YOLO 实际标注')]:
            row=self.frames.get(key)
            if not row:continue
            msg,received=row;stamp=msg.header.stamp.to_nsec();age=time.monotonic()-received
            self.captions[key].set('%s | 仿真时刻 %.3f 秒 | 接收 %.1f 秒前%s'%(title,stamp/1e9,age,'（已过期）' if age>2 else ''))
            if self.rendered.get(key)!=stamp:
                array=self.bridge.imgmsg_to_cv2(msg,'bgr8');self.photo(key,Image.fromarray(cv2.cvtColor(array,cv2.COLOR_BGR2RGB)));self.rendered[key]=stamp
        for item in self.table.get_children():self.table.delete(item)
        annotated=self.frames.get('yolo');same=annotated and abs(annotated[0].header.stamp.to_sec()-self.scene.get('stamp',-1))<1e-6
        if same:
            for c in self.scene.get('candidates',[]):
                self.table.insert('', 'end',values=(c['id'],NAMES.get(c['class'],c['class']),'%.1f%%'%(100*c['confidence']),str(c['pixel']),c.get('body_axis_deg','')))
        record=copy.deepcopy(self.requests.get(self.latest_request,{}));source=record.get('input',{})
        raw=record.get('model_answer',{}).get('answer','尚无模型输出')
        instruction=source.get('instruction','等待实际指令');decision=record.get('plan_published',record.get('dry_run_validated',{}))
        selected=decision.get('plan',{}).get('model_selection')
        rejection=record.get('rejected',record.get('validation_failed',{})).get('reason')
        self.model_picture(record)
        self.captions['model'].set('九格实际输入（固定对应本次请求） | '+str(self.latest_request or '等待请求'))
        overview='完整原始指令\n'+instruction+'\n\n九格原始输出\n'+raw+'\n\n'
        overview+='该调用只返回选择结果；没有返回中间推理文字。\n\n' if raw.lstrip().startswith('{') else ''
        overview+='程序校验\n'+('已通过，真实发布的选择：'+json.dumps(selected,ensure_ascii=False) if selected else str(rejection or '等待校验'))
        overview+='\n\n实际执行阶段\n'+'\n'.join(list(self.events)[-12:])
        if self.preview_reasons:overview+='\n\n规划预检原因与次数\n'+json.dumps(dict(self.preview_reasons),ensure_ascii=False,indent=2)
        self.text(self.answer,overview);self.text(self.prompt,'九格实际输入文本（完整提示和候选，不补写推理）：\n\n'+source.get('prompt','等待实际输入'))
        self.text(self.timeline,'实际事件记录\n'+'\n'.join(self.events))
        transport=(self.scene or {}).get('transport_status') or {}
        questions=('传送带观察器状态：'+str(transport.get('state','尚未上报'))
                   +('（release '+str(transport.get('release_id'))+'）' if transport.get('release_id') else '')
                   +'\n放置核验要它认出一个"新出现在带上"的物件；'
                   +'它若长时间停在"等工具让开"或"等稳定运动"，就是没认出来。\n\n')
        questions+='请核对：\n1. 编号对象的类别是否正确？\n2. 所选对象是否满足原指令的方位与数量？\n3. 放置的左右传送带是否正确？\n4. 抓取点是否在可夹持部位，是否有遮挡？\n\n当前原文：'+instruction

        if self.grasp:
            questions+='\n\n实际生成的抓取候选（这些是规划候选，不等于已经夹取）：\n'
            for number,item in enumerate(self.grasp['generated']['candidates']):
                c=item['pose_candidate'];questions+='C%d  %s  像素%s  夹取角度%s度\n'%(number,item['candidate_id'],c.get('pixel'),c.get('angle_deg'))
            self.grasp_picture()
        if any('NoneType' in str(k) for k in self.preview_reasons):questions+='\n本次规划遇到程序异常，尚未证实姿态不可达；这部分由程序修复，不需要用图像判断来代替。'
        self.text(self.questions,questions)
        self.context={'request_id':self.latest_request,'instruction':instruction,'model_input_image':source.get('image'),
                      'frame_id':source.get('frame_id'),'raw_answer':raw,'selected':selected,'displayed_at':time.time()}

    def tick(self):
        try:
            with self.lock:
                self.reload_config();summary={};path=Path(self.config.get('episode_dir','/missing'))/'summary.json'
                if path.exists():summary=json.loads(path.read_text())
                batch='30轮：尚无批量完成记录'
                campaign_path=self.config.get('campaign_file')
                if campaign_path and Path(campaign_path).exists():
                    campaign=json.loads(Path(campaign_path).read_text());rows=campaign.get('results',[])
                    batch='批量测试：已记录%d/%s轮，成功%d轮'%(len(rows),campaign.get('planned_rounds','?'),sum(row.get('status')=='succeeded' for row in rows))
                self.header.set('当前：%s | 已确认放置 %s 件，待核验 %s 件 | 近期停止回执：%s | %s'%(self.config.get('label','待配置'),max(summary.get('verified_objects',0),len(self.verified_releases)),len(self.pending_releases),self.stop.get('state','等待反馈') if time.monotonic()-self.stop_received<2. else '暂无本轮新回执',batch))
                if not self.frozen:self.render()
        except Exception as error:self.feedback_status.set('显示更新错误：'+str(error))
        self.root.after(300,self.tick)

    def toggle(self):
        self.frozen=not self.frozen;self.freeze_button.configure(text='恢复实时显示' if self.frozen else '固定当前画面以便核对')
        self.feedback_status.set('画面与九格记录已固定；机械臂状态仍单独更新。' if self.frozen else '已恢复实时显示；意见只作为开发核对记录。')

    def show_review_image(self):
        try:
            path=Path(self.config.get('review_image','/missing')).resolve()
            allowed=Path(os.environ['TCEI_RUNS']).resolve()
            if allowed not in path.parents or not path.is_file():raise ValueError('尚未配置本次要核对的原图')
            self.pictures['上轮放置原图']=Image.open(path).convert('RGB')
            self.full_image('上轮放置原图')
        except Exception as error:self.feedback_status.set('原图读取失败：'+str(error))

    def feedback(self,verdict):
        row={'time':time.time(),'verdict':verdict,'note':self.note.get(),'context':copy.deepcopy(self.context),'source':'user_clicked_remote_review_window','control_commands_sent':False}
        with (self.out/'human_feedback.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        self.feedback_status.set('已记录：'+verdict+'。没有向机械臂发出指令。')

    def close(self):
        for sub in self.subs:sub.unregister()
        rospy.signal_shutdown('review window closed');self.root.destroy()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    window=Review(args.config,args.output)
    try:window.root.mainloop()
    finally:(Path(args.output)/'closed.json').write_text(json.dumps({'time':time.time(),'control_publishers':0}))
