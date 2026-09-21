# RoboticArm

TCEI 2026 无人系统具身智能算法挑战赛 —— 机械臂物资分拣场景应用挑战赛。

用自然语言指令驱动 JAKA 机械臂，在 Isaac Sim 仿真中把物资分拣到左右两条传送带。

```
你说一句话 →九格大模型挑出目标 →YOLO+深度相机算位置 →程序规划动作 →机械臂执行
```

---

## 目录结构

```
RoboticArm/
├── arm/       机械臂运行层 —— 要部署到服务器的那套代码
├── records/   实验记录 —— 压力测试、提速测试、专项验证的结论文件
└── local/     本地工作区 —— 官方源码、文档、笔记、脚本
```

### `records/` —— 实验记录

实验与验证的结论性文件（纯文本，约 8M），按日期分组：

| 目录 | 说明 |
| --- | --- |
| `20260918_stress/` | 30 轮压力测试：审计结论、版本冻结、开发脚本 |
| `20260918_speed/` | 提速测试：基线审计、最终状态 |
| `20260919_competition/` | 实施与专项验证：任务执行记录、探针脚本、补丁、诊断报告 |
| `deploy/` | 部署实例记录 |

**原始运行数据不在仓库里**（RGB-D、事件流、视频、模型权重），详见 `records/README.md`。

### `arm/` —— 运行层（部署用）

自包含、可直接运行。服务器 clone 完 `cd arm` 就能跑。

| 内容 | 说明 |
| --- | --- |
| `robot.sh` | 统一入口：`init / doctor / start / run / status / cancel / stop / panel / desktop` |
| `tcei_stack/` | 34 个运行模块（感知、九格接口、决策、抓取、证据、停止） |
| `scripts/` | 校验、初始化、环境自检、启动、停止等入口脚本 |
| `test_tools/` | 批量测试入口 |
| `harness/` | 测试支架 |
| `config/` | 镜像基线清单 |
| `calibration/` | 相机/机器人投影标定 |
| `evaluation/` | 深度归档等评估工具 |
| `cases_b01/` | 原 30 轮场景 |
| `cases_official30_v2/` | 新 30 轮场景 |
| `instructions/` | 官方指令原文 |
| `validation/` | 验证记录 |
| `docs/` | 部署说明、测试计划、修改计划、轨迹调优等 |
| `PACKAGE_MANIFEST.json` | 文件清单与校验值 |

### `local/` —— 本地工作区（不分发到服务器）

| 内容 | 说明 |
| --- | --- |
| `official/` | 官方比赛源代码（EAICON / jaka / inference），只读基线 |
| `source_code/` | 早期留存的源码副本（含编译产物，仅供追溯） |
| `docs/` | 说明文档 |
| `notes/` | 学习笔记 |
| `scripts/` | 本地工具（云服务器操控脚本等） |
| `logs/` | 本地日志与截图 |

---

## 服务器上怎么拉代码

### 前置条件

必须用**官方赛事镜像**开的实例。运行层依赖镜像里已经装好的资源，路径是固定的：

| 资源 | 镜像内路径 |
| --- | --- |
| Isaac Sim 4.5 | `/root/isaacsim/python.sh` |
| 官方场景与控制基础 | `/root/EAICON/` |
| YOLO 权重 | `/root/jaka/best.pt` |
| ROS Noetic 工作空间 | `/opt/ros/noetic/` + `/root/jaka/devel/setup.bash` |
| 九格模型（约 4B） | `/root/inference/FM9G4B-V` |
| 第三方库 | `/root/ultralytics/` |

这些**不在仓库里**（体积太大），由镜像提供。换镜像后应重新跑一遍 `doctor`。

### 拉取

在 JupyterLab 的终端里：

```bash
cd /root
git clone https://gitee.com/learning_Z/RoboticArm.git
cd RoboticArm/arm
```

如果是私有仓库，需要先配置凭据（用户名 + 访问令牌，或 SSH key）：

```bash
# 方式一：令牌（推荐）
git clone https://<用户名>:<访问令牌>@gitee.com/learning_Z/RoboticArm.git

# 方式二：SSH
git clone git@gitee.com:learning_Z/RoboticArm.git
```

### 更新已有副本

```bash
cd /root/RoboticArm
git pull
```

拉完建议重新校验一次：

```bash
cd arm && /usr/bin/python3 scripts/verify_package.py
```

---

## 怎么跑

全部命令都在 `arm/` 目录下执行。

```bash
cd /root/RoboticArm/arm

# 1. 校验包完整（147 个文件，零 mismatch）
/usr/bin/python3 scripts/verify_package.py

# 2. 初始化（生成实例自己的身份标记和数据目录）
bash robot.sh init --display :20

# 3. 环境自检（官方资源、ROS/图像库、两个模型环境、桌面）
bash robot.sh doctor
```

`init` 会自己找桌面显示号；如果找不唯一，看终端的 `echo "$DISPLAY"` 再明确传入，例如 `--display :20`。

默认数据保存到 `/root/gpufree-data/tcei_260920v2`。**注意：`/root/gpufree-data/` 下的内容在释放实例、制作镜像时不保存**，重要证据要放系统盘或及时导出。

也可以指定数据目录：

```bash
bash robot.sh init --display :20 --data-root /root/gpufree-data/你的目录
```

### 跑一轮任务

```bash
# 启动仿真（开 Isaac、预热 YOLO 和九格，执行开关保持关闭）
bash robot.sh start <栈名>

# 看核对面板（可选）
bash robot.sh panel --episode <轮次名>

# 下发任务（这里以官方五条示例为例）
bash robot.sh run <轮次名> --official-example

# 查看状态 / 停止
bash robot.sh status
bash robot.sh stop
```

每次冷启动和每轮任务**用新的名称**，程序会拒绝复用已开始过的场景。

### 官方五条示例指令

1. 抓取左上方的烟雾弹，放到左侧传送带
2. 抓取右下方的弹夹，放到右侧传送带
3. 抓取最左方的军用手电筒，放到左侧传送带
4. 抓取手雷，放到右侧传送带
5. 抓取剩余的物品，放到左侧传送带

---

## 跑完要看什么

每轮目录会保留：

```
<数据目录>/<轮次名>/
├── episode/events.jsonl     完整事件流
├── episode/summary.json     本轮汇总
├── rgbd/                    RGB-D 记录
└── before_scene.json        初始场景
```

重点看 `summary.json` 里的 `status`、`verified_objects`、`elapsed_seconds`。

**两个容易误读的地方：**

1. **`completed_with_unverified_placements` 不等于成功。** 它表示释放动作都做完了，但放置证据有缺口。不能算成功，也不能因为"篮子空了"就加分。这种情况下 `run` 会返回退出码 1 —— **这是设计好的，不是崩溃。**
2. **"待核验"是诚实的中间状态**，不是失败。系统只是没有拍到物资确实落在传送带上的连贯画面，所以没有硬说成功。

---

## 一些注意事项

- **不要修改官方资源。** 运行层是用"包一层"的方式接入官方 `jaka_sim.py` 的，官方文件保持原样。直接跑官方原入口不会带上本项目的改进。
- **不要改仿真运行速度**（比如让机械臂瞬移），这是比赛明确禁止的。
- **九格大模型必须真实参与语义解析**，比赛规则写明未使用直接判 0 分。
- **失败了不要换种子重跑。** 失败要算进分母，改代码就另起一个新批次。
- **仿真通过不等于实机能过。** 急停和动力学还得实机验证。
