# 模型补丁 `modeling_fm9gv.py` 说明（含位置）

> 核查时间：2026-10-03（v1 会话）
> 来源：`arm/docs/模型与环境恢复/`（仓库内，已入库）+ `9.29存档.zip` → `存档/还原点_20260929/模型补丁/`（两份，逐字节一致）
> 关联：`训练产出-382MB适配器说明.md`（适配器/训练）、`arm/docs/模型与环境恢复/README-恢复要点.md`（恢复流程）

---

## 一、一句话

**补丁 = 打在基座模型自带的一个 Python 文件上的 3 行修改**，用来修一个会让模型**直接起不来**的报错（`TypeError: ... got multiple values for keyword argument 'input_ids'`）。它改的是**代码**，不是权重——这和 382 MB 适配器（改的是权重增量）是完全不同的两回事。

---

## 二、位置（先答这个）

补丁的"本体"是基座模型目录里的一个文件，**打补丁就是就地改它**：

| 项 | 路径 |
| --- | --- |
| **实例上被修改的文件** | **`/root/inference/FM9G4B-V/modeling_fm9gv.py`** |
| 实例上的原版备份 | `/root/inference/FM9G4B-V/modeling_fm9gv.py.bak_prepatch` |
| 同目录相关文件 | `config.json`、`configuration_fm9g.py`、`modeling_fm9g.py`、`image_processing_fm9gv.py`、`processing_fm9gv.py`、`resampler.py`、`tokenization_fm9g*.py` |

**为什么是这里**：FM9G4B-V 是**自带建模代码**的模型（`config.json` 里 `auto_map` 把 `AutoModel` 指向 `modeling_fm9gv.FM9GV`）。`nine_node.py` 用 `AutoModel.from_pretrained(path, trust_remote_code=True)` 加载时，transformers 会**直接执行这个目录里的 `modeling_fm9gv.py`**——所以补丁必须打在模型目录里，改别处无效。

补丁的**记录/副本**同时存在三处（同一份内容）：

| # | 位置 | 内容 |
| --- | --- | --- |
| ① | **仓库** `arm/docs/模型与环境恢复/` | `模型补丁-modeling_fm9gv.diff`（补丁全文）+ `apply_patch.sh`（一键脚本）+ `modeling_fm9gv.py.bak_prepatch`（原版备份）+ `config.json` |
| ② | **9.29 存档** `9.29存档.zip` → `存档/还原点_20260929/模型补丁/` | 同上一套（`modeling_fm9gv.py`、`.bak_prepatch`、`apply_patch.sh`、`config.json`） |
| ③ | **实例**（生效中） | `/root/inference/FM9G4B-V/modeling_fm9gv.py`（已打补丁，15877 B） |

> ✅ **补丁是三层资产里"已入库"的那一层**——不像 8.5GB 基座和 382MB 适配器进不了 git，补丁是**文本**（±144 B / 1 KB 级），完全适合版本控制。

---

## 三、为什么需要它（症状与根因）

**症状**：不打补丁，加载模型后一跑前向就报

```
TypeError: FM9GForCausalLM(...) got multiple values for keyword argument 'input_ids'
```

**根因**（`arm/docs/模型与环境恢复/README-恢复要点.md` 与训练脚本注释都有写）：

- FM9GV 的 `forward` 会把 `input_ids` / `position_ids` 硬写死（实际走的是 `inputs_embeds`，`input_ids` 恒为 `None`），然后把 `**kwargs` 透传给内部的 LLM。
- 4bit 量化模型会被 accelerate 挂上 `AlignDevicesHook`，它**按签名重建参数时会把 batch 里的 `input_ids` 重新塞回 kwargs**。
- 于是内部 LLM 同时收到"硬写死的 `input_ids=None`"和"kwargs 里的 `input_ids`" → **重复传参报错**。

**关键**：这个报错只在 **4bit/量化路径**下触发（`load_in_4bit`），所以它是"环境改动"的一部分——换实例装好 bitsandbytes 后**必打**。

---

## 四、补丁做了什么（diff 原文）

`模型补丁-modeling_fm9gv.diff` 全文：

```diff
--- a/modeling_fm9gv.py.bak_prepatch
+++ b/modeling_fm9gv.py
@@ -174,6 +174,9 @@
         if position_ids.dtype != torch.int64:
             position_ids = position_ids.long()
 
+        kwargs.pop("_input_ids_guard", None)
+        for _k in ("input_ids", "position_ids", "inputs_embeds"):
+            kwargs.pop(_k, None)
         return self.llm(
             input_ids=None,
             position_ids=position_ids,
```

**逐行读**：

- 位置：`modeling_fm9gv.py` 第 174 行附近，**`return self.llm(...)` 之前**。
- `kwargs.pop("_input_ids_guard", None)` — 顺手清掉一个内部标记键（防御性）。
- `for _k in ("input_ids", "position_ids", "inputs_embeds"): kwargs.pop(_k, None)` — **把这三个键从 kwargs 里摘掉**。理由（注释原文）：它们本来就该由 FM9GV 依据 `data` 自己生成，不该从外部传进来。摘掉后，内部 LLM 只收到一份 `input_ids=None`，冲突消失。

改动**净增 3 行、144 字节**，逻辑上只是"删掉误传进来的参数"，不动任何模型行为。

---

## 五、校验（怎么确认"我传过去的字节是对的"）

| 文件 | 字节数 | sha256（前 16 位） |
| --- | ---: | --- |
| `modeling_fm9gv.py`（**已打补丁**） | **15,877** | **`b6f28878b425ea6b…`** |
| `modeling_fm9gv.py.bak_prepatch`（**原版**） | **15,733** | **`50f1d9a74737b7bd…`** |

快速判断有没有打上：

```bash
grep -c _input_ids_guard /root/inference/FM9G4B-V/modeling_fm9gv.py    # ≥1 表示已打
```

---

## 六、怎么打（`apply_patch.sh` 做的事）

`arm/docs/模型与环境恢复/apply_patch.sh` 一共四步：

1. 检查 `$P = /root/inference/FM9G4B-V/modeling_fm9gv.py` 存在；
2. **备份原版**（`cp -n` 到 `.bak_prepatch`，已存在则不覆盖）+ **幂等检查**（已含 `_input_ids_guard` 就跳过）；
3. 用 Python 在 `return self.llm(` 前**插入那 3 行**（`assert` 防止重复插入）；
4. **清 transformers 远程代码缓存**：`rm -rf ~/.cache/huggingface/modules/transformers_modules`——**这一步不能省**，否则 transformers 会继续用缓存里的**旧版**代码，补丁看起来打了却不生效。最后打印 `PATCHED ✓`。

用法：`bash apply_patch.sh`。

> ⚠️ **第 4 步是踩过的坑**：`trust_remote_code` 的代码会被缓存到 `~/.cache/huggingface/modules/`，改完源文件不清缓存 = 白改。

---

## 七、它和 382 MB 适配器的区别（别混）

| | **补丁 `modeling_fm9gv.py`** | **适配器 `adapter_model.safetensors`** |
| --- | --- | --- |
| 改的是什么 | **代码**（Python 建模逻辑） | **权重**（LoRA 增量） |
| 体量 | 15,877 B（±144 B） | 382,337,688 B |
| 谁改的 | 2026-09-25 **人工**（修报错） | 9/29 **训练**产出 |
| 生效方式 | 模型加载时执行该 .py | `PeftModel` 运行时叠加到基座 |
| 入库 | ✅ **已入 git**（文本） | ❌ 不入 git（超限二进制，见适配器文档） |
| 缺了会怎样 | **模型直接报错起不来** | 分类/出框功能失效（选块仍可跑） |

---

## 八、在"三层资产"里的位置

```
基座 FM9G4B-V  (pytorch_model.bin 9.03GB, mtime 2026-03-09, 未变)     ← 不进 git
   ＋ 补丁 modeling_fm9gv.py  (+144B, 2026-09-25)  ← ★本文主角，已入库
   ＋ 适配器 adapter_model.safetensors  (382MB, 9/29)               ← 不进 git
   ＝ 可运行、且带九格识别能力的完整模型
```

补丁是三层里**唯一可以、也已经**放进 git 的一层。

---

## 九、一句话总结

**补丁 = 打在 `/root/inference/FM9G4B-V/modeling_fm9gv.py`（模型自带、被 `trust_remote_code` 直接执行的建模代码）上的 3 行修改：在 `return self.llm(...)` 前弹出 `input_ids/position_ids/inputs_embeds`，修掉 4bit 量化下的重复传参 `TypeError`。改后 15877 B / sha `b6f28878…`，原版 15733 B / sha `50f1d9a7…`。它改的是代码不是权重，且已随仓库入库（`arm/docs/模型与环境恢复/`）；打完必须清 `~/.cache/huggingface/modules/transformers_modules`。**
