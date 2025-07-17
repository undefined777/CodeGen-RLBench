#!/usr/bin/env python3
"""
Tensorboard 集成功能使用示例

新增的 Tensorboard 功能包括：
1. 实时监控训练指标
2. 可视化PPO训练过程
3. 代码质量分析图表
4. 灵活的日志控制
"""

# 🔧 基本使用示例
basic_usage_cmd = """
# 启用Tensorboard（默认启用）
python optimized_rl_trainer.py \\
  --source_lang java \\
  --target_lang cpp \\
  --model_path ~/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280/ \\
  --data_path data \\
  --output_path ./ppo_training_output \\
  --train_batch_size 8 \\
  --use_tensorboard
"""

# 🔧 自定义Tensorboard配置
custom_tensorboard_cmd = """
# 自定义Tensorboard日志目录和记录频率
python optimized_rl_trainer.py \\
  --source_lang java \\
  --target_lang cpp \\
  --model_path ~/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280/ \\
  --data_path data \\
  --output_path ./ppo_training_output \\
  --train_batch_size 8 \\
  --tensorboard_log_dir ./custom_tb_logs \\
  --log_every_n_steps 5
"""

# 🔧 禁用Tensorboard
no_tensorboard_cmd = """
# 禁用Tensorboard（节省资源）
python optimized_rl_trainer.py \\
  --source_lang java \\
  --target_lang cpp \\
  --model_path ~/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280/ \\
  --data_path data \\
  --output_path ./ppo_training_output \\
  --train_batch_size 8 \\
  --no_tensorboard
"""

# 🔧 启动Tensorboard服务器
tensorboard_server_cmd = """
# 训练开始后，在另一个终端启动Tensorboard服务器
tensorboard --logdir=./ppo_training_output/tensorboard --port=6006

# 然后在浏览器中访问：http://localhost:6006
"""

print("📊 Tensorboard 集成功能使用指南")
print("=" * 70)

print("\n🔧 新增参数说明:")
print("--use_tensorboard      : 启用Tensorboard日志记录（默认启用）")
print("--no_tensorboard       : 禁用Tensorboard日志记录")
print("--tensorboard_log_dir  : 自定义Tensorboard日志目录")
print("--log_every_n_steps N  : 每N个训练步骤记录一次指标（默认每步都记录）")

print("\n📈 监控的指标包括:")

print("\n🎯 训练指标 (Training):")
print("- Average_Reward        : 平均奖励值")
print("- Compilation_Success_Rate : 编译成功率") 
print("- AST_Match_Score       : AST匹配分数")
print("- DFG_Match_Score       : DFG匹配分数")
print("- Learning_Rate         : 当前学习率")

print("\n🔍 代码质量指标 (Code_Quality):")
print("- Avg_Errors           : 生成代码平均错误数")
print("- Avg_Errors_Ref       : 参考代码平均错误数")
print("- Avg_Nodes            : 生成代码平均节点数")
print("- Avg_Nodes_Ref        : 参考代码平均节点数")

print("\n🤖 PPO算法指标 (PPO):")
print("- KL_Divergence        : KL散度")
print("- Entropy              : 策略熵")
print("- Total_Loss           : 总损失")
print("- Policy_Loss          : 策略损失")
print("- Value_Loss           : 价值函数损失")
print("- Advantages_Mean      : 优势函数均值")
print("- Returns_Mean         : 回报均值")
print("- Value_Mean           : 价值函数均值")

print("\n📊 评估指标 (Evaluation):")
print("- Train_Errors         : 训练集错误数")
print("- Test_Errors          : 测试集错误数")
print("- Train_Error_Rate     : 训练集错误率")
print("- Test_Error_Rate      : 测试集错误率")

print("\n🚀 使用示例:")

print("\n📝 基本使用（推荐）:")
print("启用Tensorboard，使用默认设置")
print(basic_usage_cmd)

print("\n📝 自定义配置:")
print("自定义日志目录，每5步记录一次指标")
print(custom_tensorboard_cmd)

print("\n📝 禁用Tensorboard:")
print("完全禁用Tensorboard以节省资源")
print(no_tensorboard_cmd)

print("\n🌐 启动Tensorboard服务器:")
print("在训练过程中或训练后查看图表")
print(tensorboard_server_cmd)

print("\n💡 使用技巧:")
print("1. 📊 实时监控: 训练时同时启动Tensorboard服务器，实时查看指标变化")
print("2. 🔍 对比实验: 使用不同的run_id进行多次实验，在Tensorboard中对比结果")
print("3. 📈 性能分析: 关注AST_Match_Score和DFG_Match_Score的变化趋势")
print("4. 🎯 调参指导: 根据KL_Divergence和Policy_Loss调整学习率和KL系数")
print("5. 💾 资源优化: 长时间训练时可设置较大的log_every_n_steps以减少IO开销")

print("\n🎯 推荐配置:")
print("对于你的 Java-to-C++ 训练:")
print("  --use_tensorboard           # 启用监控")
print("  --log_every_n_steps 2       # 每2步记录一次，平衡细节和性能")
print("  --tensorboard_log_dir ./tb_logs  # 自定义日志目录")

print("\n📁 日志目录结构:")
print("./ppo_training_output/tensorboard/")
print("├── run_1_20240117_143022/")
print("│   ├── events.out.tfevents.xxx")
print("│   └── ...")
print("└── run_2_20240117_150045/")
print("    ├── events.out.tfevents.xxx")
print("    └── ...")

if __name__ == "__main__":
    print("\n🎉 Tensorboard集成完成！")
    print("现在你可以：")
    print("✅ 实时监控PPO训练过程")
    print("✅ 可视化代码质量指标")
    print("✅ 分析AST匹配改进效果")
    print("✅ 对比不同实验的结果") 