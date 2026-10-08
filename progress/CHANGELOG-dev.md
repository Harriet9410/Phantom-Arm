# 进度日志（log）

> 规则：每次本地改动按阶段追加；**本阶段完成后不 push**。  
> 源码根：`D:\MYCODE\TCEI\source_code`  
> 远程（当前）：`https://github.com/learnerCodeZ/Phantom-Arm`（`main` / `048c2e0` 起点）

---

## 阶段 1：大模型输出契约优化（Prompt + 解析）

**状态：** 已完成（本地）  
**目标：** 统一「指令 → 模型输出 → C++ 解析」格式，减少 `/model_output` 解析失败。  
**范围：** 只改感知/指令契约，不改 ROS 话题名、不改抓取状态机、不改 Isaac。

### 改动清单（按先后）

| 顺序 | 文件 | 改了什么 |
| --- | --- | --- |
| 1 | `jaka/src/large_scale_model_arm/scripts/isaac_scale.py` | 新增 `build_user_prompt()`：用户指令包进固定输出契约 |
| 2 | 同上 | `process_latest_frame()` 改为发送契约 Prompt，不再把原始 input 直接丢给模型 |
| 3 | 同上 | 默认 prompt 改为契约向说明（仍可被终端输入覆盖） |
| 4 | `jaka/src/large_scale_model_arm/src/isaac_yolov8.cpp` | 放宽元组正则：支持 `()`/`[]`/全角`（）`、英文/中文逗号、`m` 可选、浮点坐标 |
| 5 | 同上 | 深度/角度合法性检查：非法深度跳过、异常角度告警并夹紧 |
| 6 | 同上 | 分拣侧同时识别 `left`/`right` 英文 |
| 7 | 同上 | 解析失败时的 WARN 写明期望格式 |
| 8 | `jaka/src/large_scale_model_arm/scripts/test_model_contract.py` | 本地无 ROS 单测：Prompt 组装 + 正则解析 + 侧向判断 |

### 行为变化（对外）

**改前：**

- 默认 Prompt：`请处理图像并返回结果`（无输出格式约束）
- C++ 只认：`(整数, 整数, 数字m, 数字)`，缺 `m` 或用 `[]` 就失败

**改后：**

- Prompt 固定要求：`(u, v, depth_m, angle_deg)`，可选一行 `左侧/右侧`
- C++ 更宽容地解析契约元组，并丢弃明显非法深度
- 下游话题与 7 维位姿数组格式不变

### 本地验证

```text
python test_model_contract.py → ALL PASS
```

首跑失败用例：全角括号 → 已补进 C++/Python 正则后通过。

### 未做 / 下阶段

- [ ] 云端五终端验证真实模型输出
- [ ] **不 push**

---

## 阶段 2：深度取值鲁棒（邻域采样 + 异常过滤）

**状态：** 已完成（本地）  
**目标：** 避免中心像素 NaN/0/离群导致的错误 3D 坐标。  
**范围：** 只改 `isaac_yolov8.py`；不改话题、不改外参。

### 改动清单（按先后）

| 顺序 | 文件 | 改了什么 |
| --- | --- | --- |
| 1 | `jaka/.../scripts/isaac_yolov8.py` | `DEPTH_MIN/MAX/MIN_VALID`（与 C++ 阶段1过滤对齐） |
| 2 | 同上 | `sample_depth_robust()`：5×5 邻域、过滤非有限/越界、中位数 |
| 3 | 同上 | 无效深度不发布并 `logwarn_throttle` |
| 4 | `scripts/test_depth_sample.py` | 本地单测 |

### 本地验证

```text
test_depth_sample: ALL PASS
test_model_contract: ALL PASS（回归）
```

### 未做

- [ ] 云端看深度标注是否稳定
- [ ] **不 push**

---

## 阶段 3：抓取失败重试与状态机容错（C2）

**状态：** 已完成（本地）  
**目标：** 未到位不放置、超时不卡死、失败可重试并给出原因。  
**范围：** 只改 `isaac_grasp.cpp` + 决策单测。

### 改动清单（按先后）

| 顺序 | 文件 | 改了什么 |
| --- | --- | --- |
| 1 | `jaka_arm/src/isaac_grasp.cpp` | `waitForPosition` / `waitForGripper` 超时，返回 `bool` |
| 2 | 同上 | 单次尝试失败原因 + 失败即回初始张爪 |
| 3 | 同上 | 未到位不再继续放置 |
| 4 | 同上 | `max_retries` / timeouts / `retry_delay` 参数 |
| 5 | 同上 | `arrayCallback` 重试循环，统一发布 `/pick_place_result` |
| 6 | `jaka_arm/scripts/test_grasp_retry.py` | 重试决策单测 |

### 本地验证

```text
test_grasp_retry / contract / depth → ALL PASS
```

### 未做

- [ ] 云端 `catkin_make` + 联调
- [ ] **不 push**

---

## 阶段 4：TRRT* 参数集中与扫参表（B4 准备）

**状态：** 已完成（本地）  
**目标：** 规划参数不再散落在 `plan_and_execute_trajectory` 里；云端扫参改一处。  
**范围：** 只改 `jaka_env.py` 调用方式 + 新增 `trrt_config.py`；**不改算法本体**。

### 改动清单（按先后）

| 顺序 | 文件 | 改了什么 |
| --- | --- | --- |
| 1 | `EAICON/Source/JAKA/trrt_config.py` | **新增**：`DEFAULT_TRTT`、`PRESETS(baseline/fast/stable)`、`get_trrt_params()` |
| 2 | `EAICON/Source/JAKA/jaka_env.py` | import 配置；模块级 `TRRT_PARAMS` |
| 3 | 同上 | `set_trrt_preset()` 供云端切换 |
| 4 | 同上 | `plan_and_execute_trajectory`：从 `TRRT_PARAMS` 传入 max_iter/step/radius/temp 等；duration 可配置 |
| 5 | 同上 | 规划失败日志打印当前参数 |
| 6 | `EAICON/Source/JAKA/test_trrt_config.py` | 本地单测：键齐全、拷贝隔离、非法 preset |

### 行为变化

**改前：**

```python
trrt_star_optimized(..., step_size=0.01, radius=0.5)  # 其余吃函数默认值
duration 默认 5.0 写死在签名里
```

**改后：**

- 所有 TRRT* 旋钮集中在 `trrt_config.py`
- 预设：
  - `baseline`：与原先实际调用等价（step=0.01）
  - `fast`：step=0.03, max_iter=3000
  - `stable`：step=0.008, max_iter=8000, radius=0.6
- 云端在 Python 里：`from jaka_env import set_trrt_preset; set_trrt_preset("fast")`

### 本地验证

```text
python test_trrt_config.py → ALL PASS
```

> 本阶段未跑 Isaac；算法逻辑未改，仅参数源。

### 云端扫参建议（未做）

| 实验 | preset | 看什么 |
| --- | --- | --- |
| E0 | baseline | 成功率、规划是否失败 |
| E1 | fast | 是否更抖/撞 |
| E2 | stable | 是否更慢但更稳 |
| E3 | 自定义 | 只改 `DEFAULT_TRTT` 一处 |

### 未做

- [ ] 云端三种 preset 对比
- [ ] 真碰撞 validator（当前仍 `lambda q: True`）
- [ ] **不 push**

---

## 阶段 5：抓取接近偏移参数化 + 实验记录模板（B2 + D4）

**状态：** 已完成（本地）  
**目标：** 硬编码偏移可调；云端实验有统一模板与启动参考。  
**范围：** 只改 `isaac_grasp.cpp` 参数 + 新增脚本/模板；默认值与原硬编码一致。

### 改动清单（按先后）

| 顺序 | 文件 | 改了什么 |
| --- | --- | --- |
| 1 | `jaka_arm/src/isaac_grasp.cpp` | 参数化 `approach_dx/dy`、`descend_dx/dy`、`place_lower/raise_dz`、settle 时间 |
| 2 | 同上 | 默认值对齐原硬编码：dx=0.03/0.01，dy=0.0075，dz=0.2，settle 2.0/1.5 |
| 3 | `jaka/run_five_terminals.sh` | 五终端启动参考（路径自适应 `/root` 与 `/root/gpufree-data`） |
| 4 | `local/docs/实验记录模板.md` | 云端实验表：元信息 / 成功率 / 失败原因 / 下一轮参数 |
| 5 | `jaka_arm/scripts/test_grasp_offsets.py` | 单测：参数存在、硬编码偏移已消失、默认值正确 |

### 行为变化

**改前：** 接近/下移/放置升降写死在 `doPickPlace`。  
**改后：** `rosparam` / launch 可调；不传参时与原行为相同。

### 本地验证

```text
test_grasp_offsets / grasp_retry / trrt_config → ALL PASS
```

### 未做

- [ ] 云端按模板填第一轮实验
- [ ] **不 push**

---

## 累计本地未提交

```text
 M isaac_scale.py / isaac_yolov8.py / isaac_yolov8.cpp / isaac_grasp.cpp / jaka_env.py
?? test_*.py / trrt_config.py / run_five_terminals.sh
 文档: progress/log.md · notes/阶段1–5 · local/docs/实验记录模板.md
```

## 下一阶段（未开始）

待选：上传回飞书 / 合并到 Harriet9410（等权限）/ 云端实测。

---

## 阶段 6：交接基线还原（同事控制器 → arm/）

**状态：** 已完成（本地，未 commit、未 push）
**目标：** 补上仓库 `arm/` 只有 `__pycache__`、没有源码的洞，建立可追溯的交接基线。
**范围：** 只做文件还原，不改任何逻辑；不涉及训练代码、不涉及九格接入、不涉及左右判定。

### 改动清单

| 顺序 | 内容 |
| --- | --- |
| 1 | 从 `9.29存档.zip` 解压 `还原点_20260929/控制器/tcei_final_v2_23/tcei_260920v2/`（dev_v7_t1_23 冻结版） |
| 2 | 整包还原进 `arm/`：59 个 .py / 178 个文件 / 2.2 MB，剔除 `__pycache__` 与 `.ipynb_checkpoints` |
| 3 | md5 抽验忠实性：semantics.py / controller.py / robot.sh / mission_ledger.py 与存档快照全部一致 |
| 4 | 新增 `arm/还原说明.md`（来源、版本、校验、纪律、未还原项） |

### 边界确认

- 左右方位：**未改动**（推导结论：当前相机画面约定与机械臂第一人称一致，见 `local/0929/02-执行计划书.md` 调研 1）
- 九格大模型：**未接入控制层**（nine_node 保持 `_execute:=false`，端到端九格从未跑通，属加分项范围，本轮不做）
- 训练代码/资产：**未触碰**（461MB 训练 tgz 未解压，sft_v2 / 模型补丁只读）

### 待办

- [ ] 用户确认后本地 commit + 打 tag 作交接基线
- [ ] 用户浏览器跑一次 `查改动.sh`，核实 10:59 快照后实例是否又有改动
- [ ] Step 2：launch_stack.py 增加 `--foreground` 分终端模式（纯脚本改动，见执行计划书）

---

## 阶段 7：九格分类接入控制层 + 双形态开关（阶段 2 落地）

**状态：** 已部署实例并实测（代码在实例 git 7 次提交中；本地 arm/ 同步）
**目标：** 九格（step_27000）接入控制层做分类，YOLO 降为随时可切回的兜底；双形态开关（兜底/纯九格合规）。

### 改动清单（部署于实例 tcei_final_v2_23/tcei_260920v2/tcei_stack/）

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| nine_classify.py | 新增 | NineRotationDetector（同 detect() 契约）：深度分割出块 → 分类经 ROS 服务问 nine_node（复用已加载模型） |
| perception.py | 修改 | `~detector`（yolo|nine，每帧读取支持热切换）+ `~yolo_enabled` 总闸（false 时不加载 best.pt，加分合规形态） |
| nine_node.py | 修改 | model_lock 串行化 + `/tcei/classify_request|result` 通用视觉问答服务 + expandable_segments 分配器 |
| BUILD_MANIFEST.json | 重算 | perception/nine_node 新哈希（launch_stack 校验通过） |

### 调试历程（3 个修复）

1. 热切换不生效 → detector 改为每帧读参数；
2. 分类回答带 ```json 围栏解析失败 → _parse_answer 剥壳；
3. 语义选择 CUDA OOM（1.54/2.24GiB）→ expandable_segments + empty_cache + TTL 降频。

### 实测成绩

- **语义测试（dry-run）**：五条指令两轮零语义错误（方位词/side 全对）；
- **YOLO 兜底形态回合（semdemo0930a，用户录屏）**：**succeeded 5/5 全核验**（历史最好，含"剩余的物品"首次核验通过），全程 242 秒；
- **九格分类模式**：分类服务打通（57+ 次回答），但持续分类+语义选择存在**显存压力**（扫描回答曾 200 token 截断已修；碎片化 OOM 已修）——待按需化重构（见 02 文档第七节 B 方案）。

### 遗留

- [ ] 九格分类按需化重构（B 方案）→ 纯九格 demo
- [ ] 入口 B 指令控制台
- [ ] 九格分类质量结论反馈训练侧
