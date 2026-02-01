"""
Runner for the lifelong-learning benchmark.

当前阶段只实现最小闭环（只考虑 DBBench + zero_shot + single_agent）：
- 从 configs/assignment/default.yaml 读取实验配置；
- 使用 scheduler 生成 (task, index) 调度序列；
- 调用后端 /start_sample 获取初始 messages + tools；
- 通过 zero_shot memory 机制（目前只是透传）生成 enhanced_messages；
- 使用 single_agent 执行引擎（占位实现）执行样本；
- 把 history + result 落盘，后续可供 memory 机制或分析使用。

后续会在此基础上逐步接入：
- 真实的 LLM 调用（基于 configs/llmapi/*.yaml）；
- /interact 交互循环；
- 其他 memory / execution 机制。
"""