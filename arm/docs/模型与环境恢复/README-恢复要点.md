# 模型与环境恢复（换新实例必读）

> 本目录收录**服务器独有、仓库原本没有**的两类关键资产：**九格大模型的补丁**与**Python 环境清单**。
> 背景：大模型本体（8.5GB）与 conda 环境不入 git（体积），但"我们改了什么"和"环境怎么装"必须有版本记录。
> 来源：提取自 `9.29存档.zip`（9/29 还原点）+ 2026-10-03 从实例 s8qr10bi-4q0536x8 现取，并已逐字节校验一致。

---

## 一、大模型 FM9G4B-V（8.5GB，仓库外）

| 项 | 值 |
| --- | --- |
| 位置 | `/root/inference/FM9G4B-V`（实例） |
| 来源 | `git@git.in.zhihu.com:cjm0015/minicpm-3o.git`（**内网仓，公网不可得**），基于 MiniCPM-3o |
| 内容 | `pytorch_model.bin` 9.03GB + 自定义建模代码（`modeling_fm9g.py`/`modeling_fm9gv.py`/`configuration_fm9g.py`/`image_processing_fm9gv.py`/`processing_fm9gv.py`/`resampler.py`/`tokenization_fm9g*.py`）+ `config.json` |
| 谁在用 | `tcei_stack/nine_node.py` 经 `_model_path:=/root/inference/FM9G4B-V` 加载；`nine_classify.py` 发推理请求 |
| 校验（应用补丁后） | `modeling_fm9gv.py` 应为 **15877 B / sha256 `b6f28878b425ea6b…`** |
| 校验（原版） | `modeling_fm9gv.py.bak_prepatch` 应为 **15733 B / sha256 `50f1d9a74737b7bd…`** |

> ⚠️ **模型本体没有仓库副本**。换实例/重装时必须有另一份来源（内网 git / 已有快照）。本目录只能保证**补丁与配置**可复原。

---

## 二、模型补丁（本目录核心）

- **文件**：`模型补丁-modeling_fm9gv.diff`（补丁全文）、`apply_patch.sh`（一键脚本）、`config.json`（模型配置）、`modeling_fm9gv.py.bak_prepatch`（**原版备份**）
- **症状**：不打补丁会报 `TypeError: FM9GForCausalLM(...) got multiple values for keyword argument 'input_ids'`
- **内容**（3 行）：在 `self.llm(...)` 调用前清理会重复传入的 kwargs

  ```python
  kwargs.pop("_input_ids_guard", None)
  for _k in ("input_ids", "position_ids", "inputs_embeds"):
      kwargs.pop(_k, None)
  ```

- **应用**：`bash apply_patch.sh`（脚本自带：备份原版 → 打补丁 → 清 `~/.cache/huggingface/modules/transformers_modules` → 自检打印 `PATCHED ✓`）
- **历史**：2026-09-25 在实例上直接修改；此前**没有任何版本记录**，本目录是它第一次入库

---

## 三、Python 环境（两个，均在仓库外）

| 环境 | 用途 | Python | 清单文件 |
| --- | --- | --- | --- |
| `/opt/conda/envs/inference` | 九格大模型（`nine_node.py`） | 3.10.19 | `requirements-inference.txt`（83 包） |
| `/opt/conda/envs/yolov8` | 感知（`perception.py`，零 YOLO 形态下仍用此解释器） | 3.8.20 | `requirements-yolov8.txt`（53 包） |

**关键版本（inference）**：torch 2.3.1+cu121、transformers 4.44.2、peft 0.11.1、accelerate **0.33.0**、bitsandbytes 0.46.1、numpy 1.24.4

**关键版本（yolov8）**：torch 2.4.1、torchvision 0.19.1、`-e git+…github.chenc.dev/https://github.com/ultralytics/ultralytics.git@8207609c…`、opencv-python 4.13.0.92

### ⚠️ 必须记住的四个环境改动（重装新实例时**必做**）

1. **accelerate 必须 0.33.0**（1.2.1 会让 4-bit 模型 `.to()` 报错）
2. 装 `peft` + `bitsandbytes`
3. 给 `/root/inference/FM9G4B-V/modeling_fm9gv.py` **打补丁**（见 `apply_patch.sh`）
4. 打完补丁**清缓存**：`rm -rf ~/.cache/huggingface/modules/transformers_modules`

---

## 四、换实例恢复清单（按序）

```bash
# 0. 前提：模型目录已在 /root/inference/FM9G4B-V（否则先从内网 git 取）
# 1. 建两个 conda 环境并装依赖
/opt/conda/envs/inference/bin/pip install -r requirements-inference.txt   # 或按清单手工对齐
/opt/conda/envs/yolov8/bin/pip   install -r requirements-yolov8.txt
# 2. 打模型补丁（会自动备份原版）
bash apply_patch.sh
# 3. 校验补丁生效
grep -c _input_ids_guard /root/inference/FM9G4B-V/modeling_fm9gv.py   # 期望 ≥1
# 4. 部署包 + 生成实例配置
#    git clone <本仓库> /root/tcei_final_v2_23/tcei_260920v2
#    cd 包目录 && bash robot.sh init        # 生成 state/instance.env 与 INSTANCE_ID.txt
# 5. 体检 + 冒烟
bash robot.sh doctor && bash robot.sh start test_stack_$(date +%m%d%H%M)
```

---

## 五、仍缺的（本目录不覆盖）

| 缺口 | 说明 |
| --- | --- |
| 模型本体 | 8.5GB，仅存在内网 git 与实例；建议在另一持久位置留 tar 快照 |
| Isaac Sim（15GB） | 版本号与安装方式未记录，可另行补一行 |
| `ultralytics` 源码 | yolov8 环境中是 `-e git+…github.chenc.dev/…` 形式，需该代理可达 |
| 运行数据 | 无需备份 |
