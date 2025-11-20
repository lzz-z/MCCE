# MCCE: Multi-objective Cooperative Co-Evolution with LLMs

MCCE是一个基于大语言模型（LLM）的多目标优化框架，支持分子优化、旅行商问题（MOTSP）、车辆路径问题（MOCVRP）和圆形填充问题。

## 项目特点

- **模型协同**: 支持API模型（如GPT、Claude、Gemini）与本地Qwen模型协同工作
- **DPO训练**: 集成直接偏好优化（DPO）训练，自动生成训练数据并fine-tune模型
- **多问题支持**: 支持分子优化、MOTSP、MOCVRP、圆形填充四类优化问题
- **自包含**: 所有依赖代码和数据文件都在项目内部，无需外部路径依赖

## 项目结构

```
MCCE/
├── algorithm/          # 核心算法实现
│   ├── MOO.py         # 多目标优化算法
│   ├── base.py        # 基础类定义
│   └── PromptTemplate.py  # 提示模板
├── model/             # 模型实现
│   ├── MOLLM.py       # 主模型类
│   ├── LLM.py         # LLM接口
│   └── util.py        # 工具函数
├── problem/           # 问题定义
│   ├── molecules/     # 分子优化
│   ├── motsp/         # 多目标TSP
│   ├── mocvrp/        # 多目标CVRP
│   └── circlepacking/ # 圆形填充
├── tools/             # 数据生成工具
│   ├── makerldata_dpov3.py           # 分子DPO数据
│   ├── makerldata_motsp_embed.py     # MOTSP DPO数据
│   ├── makerldata_mocvrp_embed.py    # MOCVRP DPO数据
│   └── makerldata_circle_embed.py    # 圆形填充DPO数据
├── training/          # 训练脚本
│   └── train_dpo.py   # DPO训练实现
├── data/              # 数据目录
│   ├── problems/      # 问题数据文件
│   ├── dpo_training/  # DPO训练数据（自动生成）
│   └── dpo_models/    # DPO训练模型（自动生成）
├── oracle/            # 分子评估数据
├── eval.py            # 评估模块
└── main.py            # 主入口
```

## 环境配置

本项目需要两个conda环境：

### 1. moorl环境（主执行环境）

```bash
conda create -n moorl python=3.10
conda activate moorl
pip install -r requirements_moorl.txt
```

### 2. verl环境（DPO训练环境）

```bash
conda create -n verl python=3.10
conda activate verl
pip install -r requirements_verl.txt
```

详细的环境配置说明请参考：
- `环境配置说明.md` - 详细配置步骤
- `环境安装快速指南.txt` - 快速安装指南
- `环境文件总结.md` - 环境文件说明

## 快速开始

### 运行分子优化任务

```bash
conda activate moorl
python main.py problem/molecules/config.yaml
```

### 运行MOTSP任务

```bash
conda activate moorl
python main.py problem/motsp/config.yaml
```

### 运行MOCVRP任务

```bash
conda activate moorl
python main.py problem/mocvrp/config.yaml
```

### 运行圆形填充任务

```bash
conda activate moorl
python main.py problem/circlepacking/config.yaml
```

## 配置说明

每个问题都有独立的配置文件，主要参数包括：

- `max_generation`: 最大迭代代数
- `pop_size`: 种群大小
- `model_collaboration`: 是否启用模型协同
- `use_dpo`: 是否使用DPO训练
- `model_name`: API模型名称（如 `gemini-2.5-flash-nothinking`）
- `local_model_path`: 本地Qwen模型路径

## 自定义优化问题

要定义新的优化问题，需要创建以下文件：

1. **`config.yaml`** - 算法参数配置
2. **`{problem}.yaml`** - 问题描述和目标定义
3. **`evaluator.py`** - 评估函数实现

详细的教程请参考各问题目录下的示例文件。

## DPO训练

MCCE会在优化过程中自动：
1. 收集优化数据（chosen/rejected样本对）
2. 生成DPO训练数据集
3. 启动DPO训练（使用verl环境）
4. 更新模型权重

训练数据和模型会保存在 `data/dpo_training/` 和 `data/dpo_models/` 目录下。

## 注意事项

1. 首次运行会自动下载本地Qwen模型，需要较长时间
2. DPO训练需要GPU支持，建议使用CUDA环境
3. API模型需要配置相应的API密钥
4. 优化过程中会生成大量日志和数据文件

## 引用

如果您使用了本项目，请引用相关论文。

## 许可证

本项目采用 MIT 许可证。
