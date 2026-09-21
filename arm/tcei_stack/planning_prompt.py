"""Bounded task semantics; Nine genuinely proposes every selected candidate."""
import json


def candidate_class_index(hints):
    """Complete neutral grouping; independent of task text and grasp eligibility."""
    result={}
    for candidate in hints:
        category=candidate.get('class',candidate.get('detector_hint'))
        ident=candidate.get('id')
        if not isinstance(category,str) or not isinstance(ident,str):
            raise ValueError('candidate class index requires literal class and id strings')
        result.setdefault(category,[]).append(ident)
    return result


def feedback_principle_zh(feedback):
    """Explain a rejected category, never supply the correct task answer."""
    reason=str(feedback).lower()
    if 'unknown candidate id' in reason:
        return ('所选编号不在本帧候选中。图上的数字就是候选id；请逐字使用候选id，'
                '不要添加ID、空格或示例前缀。重新看全景和完整类别索引，由你选择满足原任务的对象。')
    if 'vision/model category disagreement' in reason:
        return ('所选编号的检测类别与回答的class不一致。先从完整类别索引核对原任务请求的类别，'
                '再在该类候选中判断方位；不要把另一个类别的编号改称为请求类别。')
    if 'model class contradicts' in reason:
        return '回答的class与原任务请求类别不一致。请重新理解原文并核对全景和全部候选，不要套用示例类别。'
    if 'model side contradicts' in reason:
        return ('回答的目的侧与原任务不一致。side只能由“放到”后的传送带方向决定，'
                '只允许left或right；抓取方位和leftmost不是目的侧。')
    if any(token in reason for token in ('invalid count','quantity contradicts','count does not match','all quantity')):
        return ('重新核对原任务要求的数量与完整匹配集合。count必须是无引号整数且等于ids长度；'
                '全部也要填写实际数量，不可写all；不得擅自扩大或减少选择。')
    if 'duplicate' in reason:
        return 'JSON键和候选id不能重复。请独立重新选择正确集合并核对count，不要重复同一对象凑数量。'
    if 'compact selection requires' in reason or 'invalid model json' in reason:
        return ('严格输出合法JSON且只有class、side、count、ids四键，不附加grasp_ready等字段；'
                '不能确定时仅输出status=needs_confirmation与具体reason。')
    if any(token in reason for token in ('spatial','class-relative extreme','implicit quantity')):
        return ('所选对象未能满足原文方位或唯一性条件。请先筛类别，再按本题相机参考判断区域或极值；'
                '不能忽略限定词或因同类唯一就任取，确有歧义时返回needs_confirmation。')
    return ('上次回答未通过严格校验。请重新审读本题原文、全景、完整候选和任务上下文，'
            '保留全部条件、正确类型和四键格式；不确定时明确说明，不要猜测。')


def _example_ids(hints):
    """Fictional demonstrations never reuse any ID from the actual scene."""
    used={row.get('id') for row in hints if isinstance(row,dict)}
    result=[]
    for index in range(1,len(used)+8):
        ident='DEMO_%d'%index
        if ident not in used:result.append(ident)
        if len(result)==7:return result
    raise ValueError('could not allocate disjoint demonstration IDs')


def planning_prompt(instruction, hints, scene=None, task_context=None, feedback=None):
    scene = scene or {}
    # The ledger uses persistent identities; the model answers in image IDs.
    # Keep ALL persistent mappings in the raw audit context, never alongside
    # remaining_ids in the prompt (native pending20 copied the persistent ID).
    task_context = task_context or {}
    model_task_context = {key: task_context[key] for key in (
        'dependencies_satisfied', 'allow_reorder', 'remaining_ids', 'reserved_ids',
        'remaining_complete', 'remaining_incomplete_reasons', 'scene_frame_id',
        'scene_fresh', 'remaining_seconds', 'coverage_scope', 'frame_id')
        if key in task_context}
    if 'released_pending_stable_ids' in task_context:
        model_task_context['released_pending_count'] = len(task_context['released_pending_stable_ids'])
    if 'missing_stable_ids' in task_context:
        model_task_context['missing_object_count'] = len(task_context['missing_stable_ids'])
    context = {key: scene.get(key) for key in ('frame_id', 'stamp', 'image_size',
               'spatial_context', 'unknown_regions', 'coverage_complete', 'scene_complete',
               'model_capabilities') if key in scene}
    correction = (('本次重答需修正：' + feedback_principle_zh(feedback) + '\n'
                   '重新阅读原任务和全景；不要复述已被拒绝的答案。\n') if feedback else '')
    demo=_example_ids(hints)
    encode=lambda value:json.dumps(value,ensure_ascii=False,separators=(',',':'))
    return (
        correction +
        '用户任务原文：' + instruction + '\n'
        '你是九格规划器，必须亲自理解原任务并观察保留真实布局的全景编号图选择对象。'
        '新协议nine.selection.compact.v1：正常只输出四键JSON：class,side,count,ids。不要Markdown或额外键。'
        'count必须是JSON整数，不加引号，不写"all"，且等于ids长度。'
        'ids为字符串数组，逐字复制候选id，不加ID前缀，不重复、不编造。'
        '无法确定只输出{"status":"needs_confirmation","reason":"具体原因"}。\n'
        'class只用Magazine=弹夹/弹匣，Torch=军用手电筒/手电筒，Grenade=手雷/手榴弹，'
        'Smokegrenade=烟雾弹，CompressedFood=压缩干粮/压缩食品/压缩饼干；剩余集合用Remaining。'
        'side只用left/right，由放到哪条传送带决定，不能取抓取方位中的左右词或leftmost。'
        '全部原任务限定都要满足：先筛类别再找最左等极值；即使同类唯一，也必须满足左上/右下等区域。'
        '一个/两件按精确数量选择，全部/所有选择完整匹配集合；未写数量时必须能唯一确定目标。'
        '数量不足或不能唯一确定时拒绝；不能把全部写成字符串count。\n'
        '编号/列表/裁剪图排列不是空间顺序，裁剪图只辅助外观。'
        'camera_image的x向右、y向下；上方不是物体高度或世界坐标。'
        '按spatial_context的pixel_to_reference变换与basket_roi=[左,上,右,下]判断；'
        'basket_quadrants_v1按篮内横纵中线分象限，边界和近并列的不确定情况不得任取。'
        '这是本次显式工程约定，不代表裁判已确认。\n'
        '剩余物品必须有可信任务上下文，'
        'dependencies_satisfied和remaining_complete均为true，选择其完整remaining_ids集合且不含reserved_ids。'
        'remaining_ids和reserved_ids均为本帧图上候选id，输出ids必须使用同一种编号。'
        'released_pending_count是已释放但放置待核验的数量，它们不属于本次剩余抓取集合，不得重复抓取。'
        '前序未完成、仍有保留对象或未知区域时不得宣称剩余集合完整。'
        'grasp_ready=false或identity_status不是confirmed不能直接抓取。'
        'unknown_regions可能影响答案时需确认；unobserved_unverified_object只是失踪物体历史位置，不能限定其当前位置。'
        '看不见不等于不存在，不按缺少哪个类别猜未知物体。'
        '类别/目标不存在、数量不足、语义冲突、次序/颜色/距离/否定等不支持限定或多条复合任务，均不得忽略后执行。\n'
        '以下是虚构示例，不是本题答案，示例ID禁止用于本题；示例xy为篮内归一化图像坐标。\n'
        '例1：'+demo[0]+'烟雾弹x=.1，'+demo[1]+'手电筒x=.3，'+demo[2]+'手电筒x=.8；'
        '抓最左手电筒放右侧：'+encode({'class':'Torch','side':'right','count':1,'ids':[demo[1]]})+'\n'
        '例2：'+demo[3]+'弹夹(.8,.8)，'+demo[4]+'弹夹(.2,.2)；抓右下弹夹放左侧：'+
        encode({'class':'Magazine','side':'left','count':1,'ids':[demo[3]]})+'\n'
        '例3：前序全部完成、无未知或保留对象，完整剩余集合为'+demo[5]+'弹夹和'+demo[6]+'烟雾弹；剩余放左侧：'+
        encode({'class':'Remaining','side':'left','count':2,'ids':[demo[5],demo[6]]})+'\n'
        '下面索引覆盖全部实际类别和候选，不按本题预选，也不代表都可抓；资格和位置仍以完整候选/场景为准。\n'
        '全类别候选ID索引：'+encode(candidate_class_index(hints))+'\n'
        '可信场景上下文：' + json.dumps(context, ensure_ascii=False, separators=(',', ':')) + '\n'
        '候选提示：' + json.dumps(hints, ensure_ascii=False, separators=(',', ':')) + '\n'
        '任务上下文：' + json.dumps(model_task_context, ensure_ascii=False, separators=(',', ':')) + '\n'
        '现在只回答本题原文：'+instruction+'。仅四键选择或needs_confirmation；禁止使用示例编号、附加字段或固定答案。'
    )
