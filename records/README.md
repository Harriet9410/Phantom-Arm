# records/ —— 实验记录

这里放机械臂项目的实验记录与验证结论。都是小体积的文本文件，**不含**原始运行数据（RGB-D、视频、模型权重）。

各目录按实验日期前缀命名，便于排序。

---

## 20260918_stress/ —— 压力测试（39 个文件）

30 轮原始布局的批量压力测试。

重点文件：

| 文件 | 说明 |
| --- | --- |
| `COMPLETION_AUDIT.json` | **审计结论**：30 个场景全过、131 项单元测试通过、150 次初始正确匹配、118 次独立核验放置 |
| `FORMAL_FREEZE.json` | 冻结版本信息 |
| `V4/V5/V6_FREEZE_AND_RESUME.json` | 各版本的冻结与续跑记录 |
| `campaign_interrupted_for_planning_fix.json` | 因规划问题中断的记录 |
| `initialization_retries.json` | 初始化重试记录 |
| `development_v3*.sh` / `evaluate_rotation.py` / `inject_one_drop.py` | 开发与验证脚本 |

## 20260918_speed/ —— 提速测试（4 个文件）

| 文件 | 说明 |
| --- | --- |
| `baseline_planner_audit.json` | 规划器基线审计 |
| `final_state.json` | 最终状态 |
| `official_image_after.json` | 官方镜像校验结果 |
| `sequence_v2_completed.txt` | 序列完成时间戳 |

## deploy/ —— 部署记录

| 文件 | 说明 |
| --- | --- |
| `INSTANCE_ID.txt` | 部署实例标识 |

## 20260919_competition/ —— 实施与专项验证（477 个文件）

这一批是比赛实施阶段的工作记录，内容最杂。

主要类别：

| 类别 | 命名规律 | 说明 |
| --- | --- | --- |
| 任务执行记录 | `*_job.json` + `*_status.txt` | 每次实验的启动参数与结果状态 |
| 探针脚本 | `probe_*.py`、`analyze_*.py` | 实验期间写的一次性分析脚本 |
| 补丁 | `patch_*.diff` | 开发过程中打过的补丁 |
| 专项验证 | `H01_*`、`t1_*`、`full*`、`official*` | 抓取、观察、停止、官方用例等专项 |
| 运行时快照 | `runtime*.json.xz` | 压缩的运行配置快照 |
| 诊断报告 | `*_diagnosis*.txt`、`*_report*.txt` | 失败归因分析 |

查具体某次实验的结论，优先看 `*_status.txt` 和 `*_diagnosis.txt`。

---

## 关于原始运行数据

以下内容**不在仓库里**（体积太大，且官方镜像可复现）：

- RGB-D 记录（`*.npz`、`*.jpg`）
- 完整事件流（`events.jsonl`）
- 视频与截图（`*.png`、`*.mp4`）
- 模型权重（`*.pt`、`*.safetensors`）

这些留在服务器上，或按需要单独导出。**注意 `/root/gpufree-data/` 下的内容在释放实例、制作镜像时不保存**，重要证据要及时导出到系统盘或本地。
