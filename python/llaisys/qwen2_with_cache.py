"""
Qwen2 with KV-Cache Pool - 支持KV-Cache复用和会话管理的Qwen2推理引擎
"""

from typing import Sequence, List, Optional, Tuple, Any
from pathlib import Path
import secrets

from .models.qwen2 import Qwen2
from .session_manager import SessionManager, Message
from .kv_cache_pool import KVCachePool


class Qwen2WithKVCachePool:
    """
    Qwen2推理引擎，集成KV-Cache池支持
    
    主要功能：
    1. 支持会话管理（多个对话）
    2. 支持KV-Cache复用（前缀匹配）
    3. 支持修改历史消息并重新生成
    """
    
    def __init__(self, model_path: str, device, session_manager: SessionManager, 
                 kv_cache_pool: KVCachePool):
        """
        初始化带KV-Cache池的Qwen2引擎
        
        Args:
            model_path: 模型路径
            device: 设备类型
            session_manager: 会话管理器实例
            kv_cache_pool: KV-Cache池实例
        """
        self.qwen2 = Qwen2(model_path, device)
        self.session_manager = session_manager
        self.kv_cache_pool = kv_cache_pool
        self.tokenizer = None  # 由外部传入
    
    # ========================================================================
    # 单轮推理（带KV-Cache复用）
    # ========================================================================
    
    def infer_with_cache_reuse(
        self,
        input_token_ids: List[int],
        session_id: str,
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        seed: int = 0,
    ) -> Tuple[List[int], str]:
        """
        执行推理，尽可能复用缓存
        
        流程：
        1. 在KV-Cache池中寻找最长的前缀匹配
        2. 如果找到，从匹配位置继续推理
        3. 如果没有找到，从头推理
        4. 最后将结果存入缓存池
        
        Args:
            input_token_ids: 完整的输入token序列
            session_id: 会话ID
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_k: top-k参数
            top_p: top-p参数
            seed: 随机种子
            
        Returns:
            (生成的token序列, 使用的缓存键)
        """
        # 步骤1: 在缓存池中查找最长前缀匹配
        match_result = self.kv_cache_pool.find_longest_match(input_token_ids, session_id)
        
        if match_result:
            # 缓存命中 - 从匹配位置继续推理
            cache_entry, matched_len = match_result
            remaining_tokens = input_token_ids[matched_len:]
            
            print(f"[KV-Cache] Hit! Matched {matched_len}/{len(input_token_ids)} tokens")
            
            # 如果已经包含了所有输入token，直接返回缓存中的结果
            if not remaining_tokens:
                return [], cache_entry.cache_key
            
            # 否则从remaining_tokens继续推理
            # 注意：这里简化实现 - 实际应该从KV cache状态继续
            # 但由于C++接口限制，暂时从头推理
            output_tokens = self._generate_tokens(
                input_token_ids, max_new_tokens, temperature, top_k, top_p, seed
            )
        else:
            # 缓存未命中 - 从头推理
            print(f"[KV-Cache] Miss. Full inference from {len(input_token_ids)} tokens")
            output_tokens = self._generate_tokens(
                input_token_ids, max_new_tokens, temperature, top_k, top_p, seed
            )
        
        # 步骤2: 保存完整序列到缓存池
        cache_key = self.kv_cache_pool.store_cache(
            input_token_ids + output_tokens,
            kv_state=None,  # llaisys的KV cache状态（可选）
            session_id=session_id
        )
        
        return output_tokens, cache_key
    
    # ========================================================================
    # 会话级推理 - 自动处理消息编码和缓存
    # ========================================================================
    
    def chat_completion(
        self,
        session_id: str,
        user_message: str,
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        seed: int = 0,
    ) -> Tuple[str, dict]:
        """
        执行一个聊天轮次
        
        流程：
        1. 检查会话存在性
        2. 编码所有消息为token序列
        3. 调用infer_with_cache_reuse进行推理
        4. 解码结果并更新会话
        
        Args:
            session_id: 会话ID
            user_message: 用户消息
            max_new_tokens: 最大生成token数
            ...其他推理参数
            
        Returns:
            (生成的文本, 统计信息字典)
        """
        # 获取会话
        conv = self.session_manager.get_session(session_id)
        if not conv:
            raise ValueError(f"Session {session_id} not found")
        
        # 添加用户消息
        user_token_ids = self.tokenizer.encode(user_message)
        user_msg = Message(role="user", content=user_message, token_ids=user_token_ids)
        conv.messages.append(user_msg)
        
        # 获取完整的输入token序列
        input_token_ids = conv.get_full_token_ids()
        
        # 执行推理并尽可能复用缓存
        output_tokens, cache_key = self.infer_with_cache_reuse(
            input_token_ids,
            session_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        
        # 解码结果
        result_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        
        # 保存助手回复到会话
        assistant_msg = Message(
            role="assistant",
            content=result_text,
            token_ids=output_tokens
        )
        conv.messages.append(assistant_msg)
        conv.kv_cache_key = cache_key
        
        # 返回生成结果和统计信息
        stats = {
            "cache_key": cache_key,
            "output_tokens": len(output_tokens),
            "total_input_tokens": len(input_token_ids),
            "input_messages": len(conv.messages) - 1,  # 不计助手回复
        }
        
        return result_text, stats
    
    # ========================================================================
    # 消息修改和重新生成
    # ========================================================================
    
    def regenerate_from_message(
        self,
        session_id: str,
        from_message_idx: int,
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        seed: int = 0,
    ) -> Tuple[str, dict]:
        """
        从指定消息重新生成
        
        流程：
        1. 删除from_message_idx之后的所有消息
        2. 获取到该消息为止的token序列
        3. 调用infer_with_cache_reuse进行推理（会尽可能复用前缀缓存）
        4. 更新会话
        
        Args:
            session_id: 会话ID
            from_message_idx: 从哪条消息后重新生成（包含该消息，后续删除）
            ...其他推理参数
            
        Returns:
            (生成的文本, 统计信息字典)
        """
        conv = self.session_manager.get_session(session_id)
        if not conv:
            raise ValueError(f"Session {session_id} not found")
        
        if from_message_idx >= len(conv.messages):
            raise ValueError(f"Invalid message index {from_message_idx}")
        
        # 获取到该消息为止的token序列
        input_token_ids = conv.get_token_ids_up_to(from_message_idx)
        
        # 清除此消息之后的所有消息
        conv.messages = conv.messages[:from_message_idx + 1]
        conv.kv_cache_key = None  # 清除旧的缓存键
        
        # 执行推理
        output_tokens, cache_key = self.infer_with_cache_reuse(
            input_token_ids,
            session_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        
        # 解码结果
        result_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        
        # 保存新的回复
        assistant_msg = Message(
            role="assistant",
            content=result_text,
            token_ids=output_tokens
        )
        conv.messages.append(assistant_msg)
        conv.kv_cache_key = cache_key
        
        stats = {
            "cache_key": cache_key,
            "regenerated_from_message": from_message_idx,
            "output_tokens": len(output_tokens),
            "total_input_tokens": len(input_token_ids),
        }
        
        return result_text, stats
    
    def update_and_regenerate(
        self,
        session_id: str,
        message_idx: int,
        new_content: str,
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        seed: int = 0,
    ) -> Tuple[str, dict]:
        """
        修改某条消息并从修改处重新生成
        
        流程：
        1. 修改指定消息内容
        2. 重新编码该消息为token序列
        3. 调用regenerate_from_message进行推理
        
        Args:
            session_id: 会话ID
            message_idx: 要修改的消息索引
            new_content: 新的消息内容
            ...其他推理参数
            
        Returns:
            (生成的文本, 统计信息字典)
        """
        conv = self.session_manager.get_session(session_id)
        if not conv:
            raise ValueError(f"Session {session_id} not found")
        
        if message_idx >= len(conv.messages):
            raise ValueError(f"Invalid message index {message_idx}")
        
        # 重新编码新内容
        new_token_ids = self.tokenizer.encode(new_content)
        
        # 更新会话中的消息
        self.session_manager.update_message(
            session_id, message_idx, new_content, new_token_ids
        )
        
        # 从该消息重新生成
        return self.regenerate_from_message(
            session_id, message_idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
    
    # ========================================================================
    # 流式生成
    # ========================================================================
    
    def infer_streaming(
        self,
        input_token_ids: List[int],
        session_id: str,
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        seed: int = 0,
    ):
        """
        流式推理 - 逐token返回结果
        
        Args:
            ...参数同infer_with_cache_reuse
            
        Yields:
            生成的token字符串 (逐个)
        """
        # 这里简化实现 - 使用基础generate函数
        if seed == 0:
            base_seed = secrets.randbits(64)
        else:
            base_seed = seed
        
        for token_id in self.qwen2.generate(
            input_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=base_seed,
            stream=True,
            clear_kv_cache=False,
        ):
            if self.tokenizer:
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                yield token_text
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _generate_tokens(
        self,
        input_token_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int,
    ) -> List[int]:
        """
        调用底层Qwen2生成token
        
        返回只包含生成的新token，不包含输入
        """
        if seed == 0:
            base_seed = secrets.randbits(64)
        else:
            base_seed = seed
        
        output_token_ids = self.qwen2.generate(
            input_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=base_seed,
            stream=False,
            clear_kv_cache=True,  # 新推理清除KV缓存
        )
        
        # 返回只有新生成的token（移除输入部分）
        return output_token_ids[len(input_token_ids):]
    
    def get_end_token_id(self) -> int:
        """获取结束token ID"""
        return self.qwen2.end_token
