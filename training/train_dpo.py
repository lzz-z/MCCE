# train_dpo_custom.py - 支持自定义JSON数据集的DPO训练，使用SwanLab记录
import os
import json
import torch
import swanlab
import argparse
from datasets import Dataset, load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
import time

def load_json_dataset(json_file_path: str, split_ratio: float = 0.9):
    """
    从JSON文件加载数据集
    
    Args:
        json_file_path: JSON文件路径
        split_ratio: 训练集比例（0-1之间）
    
    Returns:
        train_dataset, eval_dataset
    """
    print(f"正在加载JSON数据集: {json_file_path}")
    
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是单个对象，转换为列表
    if isinstance(data, dict):
        data = [data]
    
    print(f"数据总数: {len(data)}")
    
    # 验证数据格式
    required_fields = ['prompt', 'chosen', 'rejected']
    valid_data = []
    
    for i, item in enumerate(data):
        if all(field in item for field in required_fields):
            valid_data.append(item)
        else:
            print(f"警告: 第{i+1}条数据缺少必要字段 {required_fields}")
    
    print(f"有效数据: {len(valid_data)}")
    
    if not valid_data:
        raise ValueError("没有找到有效的数据！请确保数据包含 'prompt', 'chosen', 'rejected' 字段")
    
    # 创建Dataset对象
    dataset = Dataset.from_list(valid_data)
    
    # 打印数据集信息
    print("数据集列名:", dataset.column_names)
    print("数据集特征:", dataset.features)
    print("\n第一条数据示例:")
    first_example = dataset[0]
    for key, value in first_example.items():
        print(f"{key}: {value}")
    
    # 分割训练集和验证集
    if len(dataset) > 1:
        split_point = int(len(dataset) * 1)
        train_dataset = dataset.select(range(split_point))
        # eval_dataset = dataset.select(range(split_point, len(dataset)))
        eval_dataset = train_dataset
        print(f"\n训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(eval_dataset)}")
        return train_dataset, eval_dataset
    else:
        print(f"\n训练集大小: {len(dataset)}")
        print("数据量太少，不分割验证集")
        return dataset, None

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="DPO training for molecular design")
    parser.add_argument("--train_data_path", required=True, help="Path to the training JSON data")
    parser.add_argument("--output_dir", required=True, help="Output directory for the trained model")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    parser.add_argument("--model_name_or_path", default="/home/lzz/models/Qwen/Qwen2.5-7B-Instruct", 
                       help="Base model path or previous trained model path")
    parser.add_argument("--ref_model_path", default="/home/lzz/models/Qwen/Qwen2.5-7B-Instruct",
                       help="Reference model path (should always be the original base model)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    # 为了避免显存不足，默认 batch size 调小；如需更大可在命令行手动覆盖
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    
    args = parser.parse_args()
    
    # 初始化SwanLab
    swanlab.init(
        project="DPO-MOLLM-Training",
        experiment_name=args.exp_name,
        description="DPO训练分子设计任务，使用MOLLM数据集",
        config={
            "model": args.model_name_or_path,
            "dataset": args.train_data_path,
            "task": "molecule_design_preference_optimization",
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "beta": args.beta
        }
    )
    
    # 训练日志输出到 JSONL（每步一条）
    train_log_dir = "/home/lzz/mollm_results/exp/gemini-2.5-flash-nothinking/train_log"
    os.makedirs(train_log_dir, exist_ok=True)
    step_log_file = os.path.join(train_log_dir, f"{args.exp_name}.jsonl")

    class StepJSONLogger(TrainerCallback):
        def __init__(self, log_path: str, exp_name: str):
            self.log_path = log_path
            self.exp_name = exp_name
            # 确保目录存在
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            record = {
                "exp_name": self.exp_name,
                "step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "timestamp": time.time(),
                "logs": logs,
            }
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"写入step日志失败: {e}")
    
    # 设置环境变量以优化内存使用
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 清理 GPU 缓存
    torch.cuda.empty_cache()

    # 配置参数
    model_name = args.model_name_or_path
    json_data_path = args.train_data_path
    output_dir = args.output_dir

    # 加载策略模型（可能是已经训练过的模型）
    print(f"正在加载策略模型: {model_name}")
    # 显式指定设备分配：策略模型放在 GPU 0-3
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="balanced",  # 使用 balanced 策略分布在多个 GPU
        max_memory={0: "14GiB", 1: "14GiB", 2: "14GiB", 3: "14GiB", 4: "14GiB", 5: "14GiB", 6: "14GiB", 7: "14GiB"},
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 加载参考模型（始终是原始基础模型）
    ref_model_name = args.ref_model_path
    print(f"正在加载参考模型: {ref_model_name}")
    # 显式指定设备分配：参考模型放在 GPU 4-7
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        torch_dtype=torch.bfloat16,
        device_map="balanced",  # 使用 balanced 策略分布在多个 GPU
        max_memory={0: "14GiB", 1: "14GiB", 2: "14GiB", 3: "14GiB", 4: "14GiB", 5: "14GiB", 6: "14GiB", 7: "14GiB"},
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 如果 tokenizer 没有 pad_token，添加一个
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    try:
        # 首先尝试加载JSON数据集
        if os.path.exists(json_data_path):
            train_dataset, eval_dataset = load_json_dataset(json_data_path)
        else:
            print(f"JSON文件不存在: {json_data_path}")
            print("使用默认的arrow数据集")
            # 回退到原来的arrow数据集
            train_dataset = load_dataset(
                'arrow',
                data_files='/home/lzz/verl_1/dataset/train/trl-lib___ultrafeedback_binarized/default/0.0.0/47124cb5778f5d50de1c7676a412828f3ea7c555/ultrafeedback_binarized-train.arrow',
                split='train'
            )
            eval_dataset = None
            
    except Exception as e:
        print(f"加载JSON数据集失败: {e}")
        print("使用默认的arrow数据集")
        train_dataset = load_dataset(
            'arrow',
            data_files='/home/lzz/verl_1/dataset/train/trl-lib___ultrafeedback_binarized/default/0.0.0/47124cb5778f5d50de1c7676a412828f3ea7c555/ultrafeedback_binarized-train.arrow',
            split='train'
        )
        eval_dataset = None

    # 配置训练参数 - 使用命令行参数配置
    training_args = DPOConfig(
        output_dir=output_dir,
        
        # 批次和梯度设置
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # 训练步数和调度
        num_train_epochs=args.num_train_epochs,
        
        # 精度和内存优化
        bf16=True,                           # 使用bf16精度（相对fp32显存占用更小）
        dataloader_pin_memory=False,         # 不使用内存固定
        remove_unused_columns=False,         # 保留所有列
        gradient_checkpointing=True,         # 启用梯度检查点
        
        # 保存和日志
        save_steps=50,                       # 保存间隔
        logging_steps=1,                     # 更频繁的日志记录
        
        # 学习率优化 - 关键调整
        # warmup_ratio=0.1,                    # 使用warmup比例而非固定步数
        warmup_steps=0,                      # 不使用warmup
        lr_scheduler_type="constant",        # 使用常数学习率
        learning_rate=args.learning_rate,    # 使用命令行参数指定的学习率
        
        # DPO特定参数优化
        beta=args.beta,                     # 使用命令行参数指定的beta值
        loss_type="sigmoid",                # 使用标准sigmoid loss
        label_smoothing=0.1,                # 添加标签平滑，提高训练稳定性
        
        # 序列长度（减小以降低显存占用）
        max_length=2048,
        max_prompt_length=1536,
        
        # 报告和监控
        report_to=["swanlab"],               # 使用SwanLab记录
        dataloader_num_workers=0,            # 单进程数据加载
        
        # 评估策略
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,  # 更频繁的评估
        
        # 保存策略
        save_strategy="steps",               # 按步数保存
        logging_strategy="steps",            # 按步数记录日志
        
        # 梯度裁剪
        max_grad_norm=1.0,                   # 梯度裁剪阈值调为1.0
        
        # 其他优化
        weight_decay=0.01,                   # 添加权重衰减
        adam_epsilon=1e-6,                   # 调整Adam epsilon
        
        # 数据处理优化
        dataloader_drop_last=True,           # 丢弃最后不完整的批次
    )

    # 创建训练器，明确指定参考模型
    step_json_logger = StepJSONLogger(step_log_file, args.exp_name)
    trainer = DPOTrainer(
        model=model, 
        ref_model=ref_model,  # 明确指定参考模型
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[step_json_logger]
    )

    # 在训练前再次清理缓存
    torch.cuda.empty_cache()

    print("开始训练...")
    trainer.train()

    print("训练完成！")
    
    # 保存模型
    trainer.save_model()
    print(f"模型已保存到: {output_dir}")
    
    # 记录最终训练指标到SwanLab
    final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    if final_metrics:
        swanlab.log({
            "final_train_loss": final_metrics.get("train_loss", 0),
            "final_rewards_accuracy": final_metrics.get("rewards/accuracies", 0),
            "final_rewards_margin": final_metrics.get("rewards/margins", 0),
        })
    
    # 结束SwanLab记录
    swanlab.finish()
    
    print("🎉 训练完成，日志已记录到SwanLab！")

if __name__ == "__main__":
    main()
