# 结合代码学习笔记 — TRRT* 参数集中配置

> 对应阶段：**阶段 4**（任务 B4 准备）  
> 关联文件：`jaka_env.py` · `trrt_config.py` · `test_trrt_config.py`  
> 上一篇：`阶段3-抓取失败重试-学习笔记.md`

---

## 1. 原先参数散落在哪

`jaka_env.py` 里有两处：

**函数默认值**（`trrt_star_optimized`）：

```python
max_iter=5000, step_size=0.1, radius=0.5,
init_temp=1.0, k0=0.1, alpha=0.95, beta=1.1
```

**调用处**（`plan_and_execute_trajectory`）：

```python
trrt_star_optimized(..., step_size=0.01, radius=0.5)
# max_iter / temp 等静默吃函数默认
```

问题：

- 真正生效的是 **0.01**，和函数注释里的 0.1 不一致  
- 想扫参要改两处或深入调用链  
- 失败时不知道当时用的哪组参数  

---

## 2. 阶段 4 做法：配置与算法分离

```text
trrt_config.py     ← 人改这里（预设/默认值）
      ↓
jaka_env.py        ← 只读 TRRT_PARAMS，调用 trrt_star_optimized
      ↓
trrt_star_optimized(...)  ← 算法本体不动
```

原则：

- **不改** RRT/TRRT 数学  
- **只改** 参数从哪来、怎么切预设  

---

## 3. 三个旋钮直觉（调参前先有画面）

| 参数 | 变大 | 变小 |
| --- | --- | --- |
| `step_size` | 规划快，路径糙、易撞 | 路径细、慢、易超时 |
| `max_iter` | 更可能找到路 | 早停、失败率↑ |
| `radius` | rewire 更积极，路径更优 | 更像朴素 RRT |
| `init_temp` / 温度 | 更敢接受长边 | 更保守 |

本仓库 `validator=lambda q: True`（**暂无真碰撞**），参数主要影响“能否到 goal + 路径点数”，不是避障质量。真碰撞打开后 `step_size` 要再调小。

---

## 4. 预设怎么选

```text
baseline  与赛前跑通一致 → 做对照
fast      抢时间：step 0.03
stable    求稳：step 0.008 + 更多 iter
```

云端切换：

```python
from jaka_env import set_trrt_preset
set_trrt_preset("fast")
```

或直接改 `trrt_config.DEFAULT_TRTT` 后重启仿真节点。

---

## 5. `duration` 为什么一起收

轨迹是 CubicSpline 在 `duration` 秒内插值：

- duration 短 → 每步关节变化大 → 跟踪易超调（阶段3 的 `NOT_IN_TOLERANCE`）  
- duration 长 → 比赛超时  

与 `step_size` 联动：路径点更密时 duration 可略减。

---

## 6. 本地单测测什么

`test_trrt_config.py`：

- 预设键齐全、step/max_iter 为正  
- `get_trrt_params` 返回拷贝（改返回值不污染源）  
- 未知 preset 报 KeyError  

**不测** 路径质量（需要 Isaac / 关节限）。

---

## 7. 云端扫参表（建议）

| 组 | preset | 记录 |
| --- | --- | --- |
| E0 | baseline | 规划成功率、耗时、是否卡 tracking |
| E1 | fast | 是否抖、是否 `APPROACH/DESCEND_TIMEOUT` |
| E2 | stable | 是否明显变慢 |
| E3 | 自定义 | 每次只动一个旋钮 |

指标对齐阶段 3 失败原因，方便和抓取成功率归因。

---

## 8. 下一步可做

1. 真碰撞 validator（接 Isaac 场景查询）  
2. 规划耗时打点（`time.perf_counter`）  
3. 把 preset 名打进日志，实验表自动可读  

---

## 9. 一句话总结

> **扫参改配置，不改算法；预设做对照，日志带参数。**  
> 阶段 4 不求更快，只求“改一处、可复现、能对比”。
