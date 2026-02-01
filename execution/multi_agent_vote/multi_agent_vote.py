from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from collections import deque

import yaml

from ..base import ExecutionEngine
from ..single_agent.single_agent import SingleAgentExecutionEngine, SingleAgentConfig


class AverageDivisionMemoryWrapper:
    """
    平均分配 memory 的包装器：为每个 agent 分配不同的经验子集。
    
    实现 average_division 策略：
    - 所有 agent 共享同一个 memory（经验存储在一起）
    - 但在 use_memory 时，每个 agent 看到不同的经验子集（平均分配）
    - 这样可以避免所有 agent 看到完全相同的经验，增加多样性
    
    对于不同的 memory 机制：
    - previous_sample_utilization: 按索引切片 deque（例如：10 个经验，3 个 agent -> agent0: [0:4], agent1: [4:8], agent2: [8:10]）
    - streamICL: 先检索更多经验，然后切片分配给每个 agent（例如：每个 agent 需要 4 个，3 个 agent -> 检索 12 个，然后分配）
    """
    
    def __init__(self, memory: Any, agent_index: int, total_agents: int):
        """
        Args:
            memory: 原始 memory 实例
            agent_index: 当前 agent 的索引（0-based）
            total_agents: agent 总数
        """
        self.memory = memory
        self.agent_index = agent_index
        self.total_agents = total_agents
    
    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为当前 agent 分配经验子集并 enhance messages"""
        # 根据 memory 类型选择不同的切片策略
        memory_type = type(self.memory).__name__
        
        if memory_type == "PreviousSampleUtilizationMemory":
            # previous_sample_utilization: 按索引切片 deque
            return self._enhance_previous_sample_utilization(task, messages)
        elif memory_type == "StreamICLMemory":
            # streamICL: 切片 RAG 检索结果
            return self._enhance_stream_icl(task, messages)
        else:
            # 其他 memory 机制：暂时不支持切片，使用全部经验
            return self.memory.use_memory(task, messages)
    
    def _enhance_previous_sample_utilization(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为 previous_sample_utilization 实现经验切片"""
        # 获取原始经验
        task_experiences = self.memory.experiences.get(task, deque())
        if not task_experiences:
            return messages
        
        # 将 deque 转换为列表以便切片
        experiences_list = list(task_experiences)
        total_experiences = len(experiences_list)
        
        if total_experiences == 0:
            return messages
        
        # 计算当前 agent 应该看到的经验范围
        # 例如：3 个 agent，10 个经验
        # agent 0: [0:4] (0, 1, 2, 3)
        # agent 1: [4:8] (4, 5, 6, 7)
        # agent 2: [8:10] (8, 9)
        chunk_size = (total_experiences + self.total_agents - 1) // self.total_agents  # 向上取整
        start_idx = self.agent_index * chunk_size
        end_idx = min(start_idx + chunk_size, total_experiences)
        
        # 切片经验
        agent_experiences = deque(experiences_list[start_idx:end_idx], maxlen=self.memory.window_size)
        
        # 创建临时 memory 实例用于格式化
        temp_memory = type(self.memory)(
            window_size=self.memory.window_size,
            sample_filter=self.memory.sample_filter,
            insertion_where=self.memory.insertion_where,
            system_prompt_template=self.memory.system_prompt_template,
            user_placeholder=self.memory.user_placeholder,
        )
        temp_memory.experiences[task] = agent_experiences
        
        # 使用临时 memory 的 use_memory
        return temp_memory.use_memory(task, messages)
    
    def _enhance_stream_icl(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为 streamICL 实现经验切片"""
        # 提取当前 question
        question = self.memory._extract_question(messages)
        if not question:
            return messages
        
        # 检索相似经验（使用更大的 top_k，然后切片）
        rag = self.memory._get_rag_for_task(task)
        if rag.insert_acc == 0:
            return messages
        
        # 检索更多经验（例如：如果每个 agent 需要 4 个，3 个 agent 总共需要 12 个）
        # 然后为当前 agent 分配一部分
        estimated_per_agent = rag.top_k
        total_retrieve = estimated_per_agent * self.total_agents
        total_retrieve = min(total_retrieve, rag.insert_acc)
        
        # 检索所有经验
        all_shots = rag.retrieve(query=question, top_k=total_retrieve)
        
        if not all_shots:
            return messages
        
        # 为当前 agent 分配经验子集
        chunk_size = (len(all_shots) + self.total_agents - 1) // self.total_agents
        start_idx = self.agent_index * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_shots))
        agent_shots = all_shots[start_idx:end_idx]
        
        if not agent_shots:
            return messages
        
        # 格式化经验文本
        fewshot_text = "\n\n\n".join(agent_shots).replace("\\", "\\\\")
        
        # 复制 messages
        enhanced = list(messages) if messages is not None else []
        
        # 根据 insertion_where 决定插入位置
        if self.memory.insertion_where == "system":
            experience_content = self.memory.system_prompt_template.format(examples=fewshot_text)
            experience_msg = {
                "role": "system",
                "content": experience_content
            }
            
            insert_idx = 0
            found_system = False
            for i, msg in enumerate(enhanced):
                if msg.get("role") == "system":
                    insert_idx = i + 1
                    found_system = True
                    break
                elif msg.get("role") == "user" and insert_idx == 0:
                    insert_idx = i
                    break
            
            enhanced.insert(insert_idx, experience_msg)
            
        elif self.memory.insertion_where == "user":
            formatted_fewshot_text = self.memory.user_placeholder.format(examples=fewshot_text)
            
            for i, msg in enumerate(enhanced):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if self.memory.user_placeholder in content:
                        enhanced[i] = {
                            **msg,
                            "content": content.replace(self.memory.user_placeholder, formatted_fewshot_text)
                        }
                    else:
                        enhanced[i] = {
                            **msg,
                            "content": formatted_fewshot_text + "\n\n" + content
                        }
                    break
        
        return enhanced
    
    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        """更新 memory（直接调用原始 memory 的 update_memory）"""
        self.memory.update_memory(task, history, result)


@dataclass
class MultiAgentVoteConfig:
    """
    Multi-agent vote 执行配置。
    
    支持：
    - agents: 多个 agent 名称列表
    - collaboration.method: 协作策略（average_division | independent_memory）
    - vote_method.method: 投票方法（consistency_vote | agent_vote）
    - vote_method.name: 投票 LLM 名称（仅用于 agent_vote）
    - vote_method.system_prompt: 投票 LLM 的 system prompt（仅用于 agent_vote）
    """
    agent_names: List[str]
    collaboration_method: str = "average_division"  # "average_division" | "independent_memory"
    vote_method: str = "consistency_vote"  # "consistency_vote" | "agent_vote"
    vote_agent_name: Optional[str] = None  # 仅用于 agent_vote
    vote_system_prompt: Optional[str] = None  # 仅用于 agent_vote


class MultiAgentVoteExecutionEngine(ExecutionEngine):
    """
    Multi-agent vote 执行引擎：多个 agent 同时执行同一样本，然后通过投票机制选择最终答案。
    
    实现两种投票方式：
    1. consistency_vote: 统计所有 agent 的操作/答案一致性，选择出现频率最高的
    2. agent_vote: 使用专门的投票 LLM 评估多个候选答案，选择最优的一个
    """
    
    def __init__(self, config: MultiAgentVoteConfig | None = None) -> None:
        self.config = config or MultiAgentVoteConfig(agent_names=["gpt-4o-mini"])
        self.K = len(self.config.agent_names)  # 多个 agent
        
        # 为每个 agent 创建 SingleAgentExecutionEngine
        self.agent_engines: List[SingleAgentExecutionEngine] = []
        for agent_name in self.config.agent_names:
            agent_cfg = SingleAgentConfig(agent_name=agent_name)
            engine = SingleAgentExecutionEngine(agent_cfg)
            self.agent_engines.append(engine)
    
    def _consistency_vote_round(
        self,
        agent_assistant_messages: List[Tuple[str, Dict[str, Any], bool, List[Dict[str, Any]]]]
    ) -> Tuple[int, Dict[str, Any], str]:
        """
        每轮一致性投票：统计所有 agent 的 assistant message，选择出现频率最高的。
        
        Args:
            agent_assistant_messages: List[(agent_name, assistant_msg, has_valid_tool_call, tool_error_msgs), ...]
        
        Returns:
            (winning_agent_index, winning_assistant_msg, winning_agent_name)
        """
        # 提取所有 agent 的 assistant message 的关键信息
        agent_actions = []
        
        for agent_name, assistant_msg, has_valid_tool_call, _ in agent_assistant_messages:
            # 提取 tool_calls 或 content 作为 action key
            action_key = None
            tool_calls = assistant_msg.get("tool_calls", [])
            if tool_calls:
                # 提取 tool_calls 的关键信息（function name + arguments）
                action_parts = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    func_args = func.get("arguments", "{}")
                    try:
                        args_dict = json.loads(func_args)
                        # 简化参数表示（只保留关键字段）
                        action_parts.append(f"{func_name}({json.dumps(args_dict, sort_keys=True)})")
                    except:
                        action_parts.append(f"{func_name}({func_args})")
                action_key = "|".join(sorted(action_parts))
            else:
                # 如果没有 tool_calls，使用 content 作为 action
                content = assistant_msg.get("content", "").strip()
                if content:
                    action_key = content[:200]  # 截断到前200字符
            
            if action_key:
                agent_actions.append((agent_name, action_key))
        
        # 统计每个 action 的出现次数
        action_counts: Dict[str, List[int]] = {}  # {action_key: [agent_indices]}
        for idx, (agent_name, action_key) in enumerate(agent_actions):
            if action_key not in action_counts:
                action_counts[action_key] = []
            action_counts[action_key].append(idx)
        
        # 选择出现次数最多的 action
        if not action_counts:
            # 如果没有有效的 action，随机选择一个 agent
            winning_idx = random.randint(0, len(agent_assistant_messages) - 1)
        else:
            # 找到出现次数最多的 action
            max_count = max(len(indices) for indices in action_counts.values())
            winning_actions = [action for action, indices in action_counts.items() if len(indices) == max_count]
            
            # 如果有多个 action 出现次数相同，随机选择一个
            winning_action = random.choice(winning_actions)
            winning_indices = action_counts[winning_action]
            
            # 从获胜的 agent 中随机选择一个
            winning_idx = random.choice(winning_indices)
        
        winning_name, winning_msg, _, _ = agent_assistant_messages[winning_idx]
        return winning_idx, winning_msg, winning_name
    
    def _agent_vote_round(
        self,
        agent_assistant_messages: List[Tuple[str, Dict[str, Any], bool, List[Dict[str, Any]]]],
        vote_agent: Any,
        current_history: List[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, Any], str]:
        """
        每轮 LLM 投票：使用专门的投票 LLM 评估多个候选 assistant message，选择最优的一个。
        
        Args:
            agent_assistant_messages: List[(agent_name, assistant_msg, has_valid_tool_call, tool_error_msgs), ...]
            vote_agent: 投票 LLM agent（SimpleHTTPChatAgent 实例）
            current_history: 当前的对话历史（用于上下文）
        
        Returns:
            (winning_agent_index, winning_assistant_msg, winning_agent_name)
        """
        # 构建候选 assistant messages 列表
        candidates = []
        for agent_name, assistant_msg, has_valid_tool_call, _ in agent_assistant_messages:
            candidate_info = {
                "agent": agent_name,
                "assistant_message": assistant_msg,
                "has_valid_tool_call": has_valid_tool_call,
            }
            candidates.append(candidate_info)
        
        # 构建投票 prompt
        candidates_text = json.dumps(candidates, ensure_ascii=False, indent=2)
        system_prompt = self.config.vote_system_prompt or "Choose the best assistant message from the following candidates: {candidates}"
        user_prompt = system_prompt.format(candidates=candidates_text)
        
        # 调用投票 LLM
        try:
            vote_messages = [
                {"role": "system", "content": "You are an expert evaluator. Analyze the candidates and choose the best one."},
                {"role": "user", "content": user_prompt}
            ]
            vote_response = vote_agent.inference(vote_messages, tools=None)
            vote_content = vote_response.get("content", "").strip()
            
            # 尝试从响应中提取选择的 agent 名称或索引
            winning_idx = 0  # 默认选择第一个
            try:
                # 尝试解析 JSON
                vote_json = json.loads(vote_content)
                if "agent" in vote_json:
                    agent_name = vote_json["agent"]
                    for idx, (name, _, _, _) in enumerate(agent_assistant_messages):
                        if name == agent_name:
                            winning_idx = idx
                            break
                elif "index" in vote_json:
                    winning_idx = int(vote_json["index"])
            except:
                # 如果不是 JSON，尝试从文本中提取 agent 名称
                for idx, (agent_name, _, _, _) in enumerate(agent_assistant_messages):
                    if agent_name.lower() in vote_content.lower():
                        winning_idx = idx
                        break
                # 如果还是找不到，尝试提取数字索引
                import re
                match = re.search(r'\b(\d+)\b', vote_content)
                if match:
                    idx = int(match.group(1))
                    if 0 <= idx < len(agent_assistant_messages):
                        winning_idx = idx
        except Exception as e:
            # 如果投票 LLM 失败，回退到随机选择
            print(f"  -> Warning: Agent vote failed: {e}, falling back to random selection")
            winning_idx = random.randint(0, len(agent_assistant_messages) - 1)
        
        winning_name, winning_msg, _, _ = agent_assistant_messages[winning_idx]
        return winning_idx, winning_msg, winning_name
    
    def _consistency_vote(
        self,
        agent_results: List[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]]
    ) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
        """
        一致性投票：统计所有 agent 的操作/答案，选择出现频率最高的。
        
        Args:
            agent_results: List[(agent_name, history, result), ...]
        
        Returns:
            (winning_agent_index, winning_history, winning_result)
        """
        # 提取所有 agent 的最终答案/操作
        # 对于 tool-calling 任务，我们比较最后的 tool_calls 或 final_answer
        agent_actions = []
        
        for agent_name, history, result in agent_results:
            # 提取最后一个 assistant 消息的 tool_calls 或 final_answer
            action_key = None
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        # 提取 tool_calls 的关键信息（function name + arguments）
                        action_parts = []
                        for tc in tool_calls:
                            func = tc.get("function", {})
                            func_name = func.get("name", "")
                            func_args = func.get("arguments", "{}")
                            try:
                                args_dict = json.loads(func_args)
                                # 简化参数表示（只保留关键字段）
                                action_parts.append(f"{func_name}({json.dumps(args_dict, sort_keys=True)})")
                            except:
                                action_parts.append(f"{func_name}({func_args})")
                        action_key = "|".join(sorted(action_parts))
                    else:
                        # 如果没有 tool_calls，使用 content 作为 action
                        content = msg.get("content", "").strip()
                        if content:
                            action_key = content[:200]  # 截断到前200字符
                    break
            
            if action_key:
                agent_actions.append((agent_name, action_key))
        
        # 统计每个 action 的出现次数
        action_counts: Dict[str, List[int]] = {}  # {action_key: [agent_indices]}
        for idx, (agent_name, action_key) in enumerate(agent_actions):
            if action_key not in action_counts:
                action_counts[action_key] = []
            action_counts[action_key].append(idx)
        
        # 选择出现次数最多的 action
        if not action_counts:
            # 如果没有有效的 action，随机选择一个 agent
            winning_idx = random.randint(0, len(agent_results) - 1)
        else:
            # 找到出现次数最多的 action
            max_count = max(len(indices) for indices in action_counts.values())
            winning_actions = [action for action, indices in action_counts.items() if len(indices) == max_count]
            
            # 如果有多个 action 出现次数相同，随机选择一个
            winning_action = random.choice(winning_actions)
            winning_indices = action_counts[winning_action]
            
            # 从获胜的 agent 中随机选择一个
            winning_idx = random.choice(winning_indices)
        
        agent_name, history, result = agent_results[winning_idx]
        
        # 在 result 中记录投票信息
        if isinstance(result, dict):
            result["vote_method"] = "consistency_vote"
            result["winning_agent"] = agent_name
            result["winning_agent_index"] = winning_idx
            result["all_agents"] = [name for name, _, _ in agent_results]
            result["action_counts"] = {action: len(indices) for action, indices in action_counts.items()}
        
        return winning_idx, history, result
    
    def _agent_vote(
        self,
        agent_results: List[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]],
        vote_agent: Any,
    ) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
        """
        LLM 投票：使用专门的投票 LLM 评估多个候选答案，选择最优的一个。
        
        Args:
            agent_results: List[(agent_name, history, result), ...]
            vote_agent: 投票 LLM agent（SimpleHTTPChatAgent 实例）
        
        Returns:
            (winning_agent_index, winning_history, winning_result)
        """
        # 构建候选答案列表
        candidates = []
        for agent_name, history, result in agent_results:
            # 提取最后一个 assistant 消息的关键信息
            candidate_info = {
                "agent": agent_name,
                "history": history[-5:] if len(history) > 5 else history,  # 只取最后5条消息
                "result": result,
            }
            candidates.append(candidate_info)
        
        # 构建投票 prompt
        candidates_text = json.dumps(candidates, ensure_ascii=False, indent=2)
        system_prompt = self.config.vote_system_prompt or "Choose the best answer from the following candidates: {candidates}"
        user_prompt = system_prompt.format(candidates=candidates_text)
        
        # 调用投票 LLM
        try:
            vote_messages = [
                {"role": "system", "content": "You are an expert evaluator. Analyze the candidates and choose the best one."},
                {"role": "user", "content": user_prompt}
            ]
            vote_response = vote_agent.inference(vote_messages, tools=None)
            vote_content = vote_response.get("content", "").strip()
            
            # 尝试从响应中提取选择的 agent 名称或索引
            # 可能的格式：{"agent": "gpt-4o-mini"} 或 "I choose agent 0" 或 "gpt-4o-mini"
            winning_idx = 0  # 默认选择第一个
            try:
                # 尝试解析 JSON
                vote_json = json.loads(vote_content)
                if "agent" in vote_json:
                    agent_name = vote_json["agent"]
                    for idx, (name, _, _) in enumerate(agent_results):
                        if name == agent_name:
                            winning_idx = idx
                            break
                elif "index" in vote_json:
                    winning_idx = int(vote_json["index"])
            except:
                # 如果不是 JSON，尝试从文本中提取 agent 名称
                for idx, (agent_name, _, _) in enumerate(agent_results):
                    if agent_name.lower() in vote_content.lower():
                        winning_idx = idx
                        break
                # 如果还是找不到，尝试提取数字索引
                import re
                match = re.search(r'\b(\d+)\b', vote_content)
                if match:
                    idx = int(match.group(1))
                    if 0 <= idx < len(agent_results):
                        winning_idx = idx
        except Exception as e:
            # 如果投票 LLM 失败，回退到随机选择
            print(f"Warning: Agent vote failed: {e}, falling back to random selection")
            winning_idx = random.randint(0, len(agent_results) - 1)
        
        agent_name, history, result = agent_results[winning_idx]
        
        # 在 result 中记录投票信息
        if isinstance(result, dict):
            result["vote_method"] = "agent_vote"
            result["winning_agent"] = agent_name
            result["winning_agent_index"] = winning_idx
            result["all_agents"] = [name for name, _, _ in agent_results]
            result["vote_response"] = vote_content if 'vote_content' in locals() else "N/A"
        
        return winning_idx, history, result
    
    def run_sample(
        self,
        task: str,
        index: int,
        session_id: int,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        agent_pool: Any,
        backend_client: Any,
        memory_for_enhance: Any = None,  # 用于 use_memory
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        执行一条样本：每轮让所有 agent 生成 assistant message，投票选择最佳行为，然后与 backend 交互。
        
        流程：
        1. 每轮循环：
           - 为每个 agent enhance messages（根据 collaboration_method）
           - 每个 agent 调用 LLM 生成 assistant message（只调用 LLM，不调用 backend）
           - 投票选择最佳的 assistant message
           - 使用选出的 assistant message 与 backend 交互
           - 将 backend 返回的结果添加到 history
           - 检查是否结束
        2. 返回完整的 history 和最终 result
        
        Args:
            agent_pool: 字典，包含每个 agent_name 对应的 SimpleHTTPChatAgent 实例
            对于 agent_vote，还需要包含 vote_agent_name 对应的 agent
        
        Returns:
            (history, result)
        """
        if not isinstance(agent_pool, dict):
            raise RuntimeError("MultiAgentVoteExecutionEngine requires agent_pool to be a dict")
        
        history = list(messages) if messages is not None else []
        current_tools = list(tools) if tools is not None else []
        
        # 记录每轮的投票信息
        vote_history: List[Dict[str, Any]] = []
        
        print(f"  -> Starting multi-agent vote: {len(self.config.agent_names)} agents, session_id={session_id}")
        
        round_num = 0
        while True:
            round_num += 1
            print(f"  -> Round {round_num}: All agents generating assistant messages...")
            
            # 1. 为每个 agent enhance messages（根据 collaboration_method）
            agent_enhanced_messages: List[Tuple[str, List[Dict[str, Any]]]] = []
            
            for agent_idx, agent_name in enumerate(self.config.agent_names):
                if agent_name not in agent_pool:
                    raise RuntimeError(
                        f"Agent '{agent_name}' not found in agent_pool. "
                        f"Available agents: {list(agent_pool.keys())}"
                    )
                
                # 根据 collaboration_method 决定如何 enhance
                if memory_for_enhance is not None:
                    if isinstance(memory_for_enhance, dict):
                        # independent_memory: 使用当前 agent 的 memory
                        enhanced_msgs = memory_for_enhance[agent_name].use_memory(task, history)
                    else:
                        # average_division: 共享 memory，但为每个 agent 分配不同的经验子集
                        memory_wrapper = AverageDivisionMemoryWrapper(
                            memory=memory_for_enhance,
                            agent_index=agent_idx,
                            total_agents=len(self.config.agent_names)
                        )
                        enhanced_msgs = memory_wrapper.use_memory(task, history)
                else:
                    enhanced_msgs = history
                
                agent_enhanced_messages.append((agent_name, enhanced_msgs))
            
            # 2. 每个 agent 调用 LLM 生成 assistant message
            agent_assistant_messages: List[Tuple[str, Dict[str, Any]]] = []
            
            for agent_name, enhanced_msgs in agent_enhanced_messages:
                agent = agent_pool[agent_name]
                
                try:
                    # 规范化 history
                    llm_history = self.agent_engines[0]._normalize_history_for_llm(enhanced_msgs)
                    
                    # 调用 LLM
                    assistant_msg = agent.inference(llm_history, current_tools)
                    
                    # 校验 tool_calls
                    assistant_msg, tool_error_msgs, has_valid_tool_call = self.agent_engines[0]._sanitize_assistant_tool_calls(assistant_msg)
                    
                    agent_assistant_messages.append((agent_name, assistant_msg, has_valid_tool_call, tool_error_msgs))
                except Exception as e:
                    print(f"  -> Warning: Agent {agent_name} LLM inference failed: {e}")
                    # 创建一个错误消息
                    error_msg: Dict[str, Any] = {
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    }
                    agent_assistant_messages.append((agent_name, error_msg, False, []))
            
            if not agent_assistant_messages:
                raise RuntimeError("All agents failed to generate assistant messages")
            
            # 3. 投票选择最佳的 assistant message
            print(f"  -> Round {round_num}: Voting on {len(agent_assistant_messages)} assistant messages...")
            
            if self.config.vote_method == "consistency_vote":
                winning_idx, winning_assistant_msg, winning_agent_name = self._consistency_vote_round(agent_assistant_messages)
            elif self.config.vote_method == "agent_vote":
                if not self.config.vote_agent_name or self.config.vote_agent_name not in agent_pool:
                    raise RuntimeError(
                        f"Vote agent '{self.config.vote_agent_name}' not found in agent_pool. "
                        f"Available agents: {list(agent_pool.keys())}"
                    )
                vote_agent = agent_pool[self.config.vote_agent_name]
                winning_idx, winning_assistant_msg, winning_agent_name = self._agent_vote_round(agent_assistant_messages, vote_agent, history)
            else:
                raise ValueError(f"Unknown vote method: {self.config.vote_method}")
            
            print(f"  -> Round {round_num}: Vote completed, winning agent = {winning_agent_name}")
            
            # 记录投票信息
            vote_history.append({
                "round": round_num,
                "winning_agent": winning_agent_name,
                "all_agents": [name for name, _, _, _ in agent_assistant_messages]
            })
            
            # 4. 获取获胜消息的详细信息（用于后续处理）
            _, _, has_valid_tool_call, tool_error_msgs = agent_assistant_messages[winning_idx]
            
            # 将选出的 assistant message 添加到 history
            history.append(winning_assistant_msg)
            
            # 如果没有合法 tool_call 且也不是明确的 Final Answer，使用 user 消息提示错误
            if not has_valid_tool_call and not self.agent_engines[0]._looks_like_final_answer(winning_assistant_msg):
                if tool_error_msgs:
                    error_content = "; ".join([msg.get("content", "") for msg in tool_error_msgs])
                    history.append({
                        "role": "user",
                        "content": f"{error_content} No valid tool calls found. Please call a tool instead. You must use one of the available tools to proceed."
                    })
                else:
                    history.append(self.agent_engines[0]._make_reprompt_user_message())
                continue
            
            # 如果有 valid tool_call，才添加 tool_error_msgs
            if tool_error_msgs:
                history.extend(tool_error_msgs)
            
            # 5. 使用选出的 assistant message 与 backend 交互
            env_out = backend_client.interact(session_id, [winning_assistant_msg])
            env_messages = env_out.get("messages", []) or []
            current_tools = env_out.get("tools", current_tools) or current_tools
            
            # 6. 追加环境产生的消息
            for m in env_messages:
                history.append(m)
            
            # 7. 检查是否结束
            status = env_out.get("status")
            finish = env_out.get("finish", status != "RUNNING")
            reward = env_out.get("reward", 0)
            metric = env_out.get("metric", {})
            
            if finish:
                result: Dict[str, Any] = {
                    "task": task,
                    "index": index,
                    "status": status,
                    "reward": reward,
                    "metric": metric,
                    "vote_method": self.config.vote_method,
                    "vote_history": vote_history,
                    "all_agents": self.config.agent_names,
                    "note": "MultiAgentVoteExecutionEngine: completed multi-turn execution via voting.",
                }
                print(f"  -> Sample completed after {round_num} rounds, reward={reward}")
                return history, result


def load_multi_agent_vote_engine_from_yaml(config_path: str) -> MultiAgentVoteExecutionEngine:
    """
    从 execution/multi_agent_vote/multi_agent_vote.yaml 读取配置，构造 MultiAgentVoteExecutionEngine。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    
    mav_cfg = raw.get("multi_agent_vote", {}) or {}
    agents_cfg = mav_cfg.get("agents", []) or []
    collaboration_cfg = mav_cfg.get("collaboration", {}) or {}
    vote_method_cfg = mav_cfg.get("vote_method", {}) or {}
    
    agent_names = [agent.get("name") for agent in agents_cfg if "name" in agent]
    if not agent_names:
        raise ValueError("No agents specified in multi_agent_vote configuration")
    
    collaboration_method = collaboration_cfg.get("method", "average_division")
    vote_method = vote_method_cfg.get("method", "consistency_vote")
    vote_agent_name = vote_method_cfg.get("name", None)
    vote_system_prompt = vote_method_cfg.get("system_prompt", None)
    
    cfg = MultiAgentVoteConfig(
        agent_names=agent_names,
        collaboration_method=collaboration_method,
        vote_method=vote_method,
        vote_agent_name=vote_agent_name,
        vote_system_prompt=vote_system_prompt,
    )
    return MultiAgentVoteExecutionEngine(cfg)

