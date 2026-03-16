"""
会话管理模块 - 支持多会话管理和消息修改重生成
"""

import uuid
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import threading


@dataclass
class Message:
    """单条消息"""
    role: str  # "system", "user", "assistant"
    content: str
    token_ids: List[int] = field(default_factory=list)  # 编码后的token序列
    timestamp: float = field(default_factory=time.time)


@dataclass
class Conversation:
    """一个完整的对话会话"""
    id: str
    title: str
    messages: List[Message] = field(default_factory=list)
    kv_cache_key: Optional[str] = None  # KV-Cache池中的关键字
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [asdict(m) for m in self.messages],
            "kv_cache_key": self.kv_cache_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }
    
    def get_full_token_ids(self) -> List[int]:
        """获取完整的token序列"""
        result = []
        for msg in self.messages:
            result.extend(msg.token_ids)
        return result
    
    def get_token_ids_up_to(self, message_idx: int) -> List[int]:
        """获取到第message_idx条消息为止的token序列"""
        result = []
        for i in range(message_idx + 1):
            if i < len(self.messages):
                result.extend(self.messages[i].token_ids)
        return result


class SessionManager:
    """管理多个对话会话，支持会话切换、消息修改、重新生成等功能"""
    
    def __init__(self, max_sessions: int = 100):
        """
        初始化会话管理器
        
        Args:
            max_sessions: 最大会话数量
        """
        self.conversations: Dict[str, Conversation] = {}
        self.active_session_id: Optional[str] = None
        self.max_sessions = max_sessions
        self._lock = threading.RLock()
    
    def create_session(self, title: Optional[str] = None) -> str:
        """
        创建新会话
        
        Args:
            title: 会话标题，如果为None则自动生成
            
        Returns:
            会话ID
        """
        with self._lock:
            if len(self.conversations) >= self.max_sessions:
                # 删除最久未使用的会话
                oldest_sid = min(self.conversations.keys(), 
                               key=lambda sid: self.conversations[sid].updated_at)
                del self.conversations[oldest_sid]
            
            session_id = str(uuid.uuid4())
            
            if title is None:
                title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            conversation = Conversation(
                id=session_id,
                title=title,
                messages=[]
            )
            
            self.conversations[session_id] = conversation
            if self.active_session_id is None:
                self.active_session_id = session_id
            
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Conversation]:
        """获取指定会话"""
        with self._lock:
            return self.conversations.get(session_id)
    
    def get_active_session(self) -> Optional[Conversation]:
        """获取当前活跃会话"""
        with self._lock:
            if self.active_session_id:
                return self.conversations.get(self.active_session_id)
            return None
    
    def switch_session(self, session_id: str) -> bool:
        """
        切换到指定会话
        
        Args:
            session_id: 目标会话ID
            
        Returns:
            成功返回True，不存在返回False
        """
        with self._lock:
            if session_id not in self.conversations:
                return False
            self.active_session_id = session_id
            return True
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话的摘要信息"""
        with self._lock:
            result = []
            for conv in self.conversations.values():
                result.append({
                    "id": conv.id,
                    "title": conv.title,
                    "message_count": len(conv.messages),
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at,
                    "is_active": conv.id == self.active_session_id,
                })
            return sorted(result, key=lambda x: x["updated_at"], reverse=True)
    
    def add_message(self, session_id: str, message: Message) -> bool:
        """
        向指定会话添加消息
        
        Args:
            session_id: 会话ID
            message: 消息对象
            
        Returns:
            成功返回True
        """
        with self._lock:
            conv = self.conversations.get(session_id)
            if not conv:
                return False
            
            conv.messages.append(message)
            conv.updated_at = time.time()
            return True
    
    def update_message(self, session_id: str, message_idx: int, 
                      new_content: str, new_token_ids: Optional[List[int]] = None) -> bool:
        """
        修改指定会话的某条消息
        
        Args:
            session_id: 会话ID
            message_idx: 消息索引
            new_content: 新的消息内容
            new_token_ids: 新的token序列（可选）
            
        Returns:
            成功返回True
        """
        with self._lock:
            conv = self.conversations.get(session_id)
            if not conv or message_idx >= len(conv.messages):
                return False
            
            msg = conv.messages[message_idx]
            msg.content = new_content
            if new_token_ids is not None:
                msg.token_ids = new_token_ids
            msg.timestamp = time.time()
            
            # 清除此消息之后的所有消息（因为输入改变了）
            conv.messages = conv.messages[:message_idx + 1]
            
            conv.updated_at = time.time()
            # 清除KV-Cache关键字，下次推理时需要重新计算
            conv.kv_cache_key = None
            
            return True
    
    def get_message_up_to(self, session_id: str, message_idx: int) -> Optional[List[Message]]:
        """
        获取指定会话中到某个消息为止的所有消息
        
        Args:
            session_id: 会话ID
            message_idx: 消息索引（包含）
            
        Returns:
            消息列表，如果会话不存在返回None
        """
        with self._lock:
            conv = self.conversations.get(session_id)
            if not conv:
                return None
            return conv.messages[:message_idx + 1]
    
    def get_messages_for_regeneration(self, session_id: str, from_idx: int) -> Optional[List[Message]]:
        """
        获取用于重新生成的消息（删除from_idx之后的所有消息）
        
        Args:
            session_id: 会话ID
            from_idx: 从哪条消息开始重新生成（相对于该消息的下一条）
            
        Returns:
            新的消息列表
        """
        with self._lock:
            conv = self.conversations.get(session_id)
            if not conv or from_idx >= len(conv.messages):
                return None
            
            # 删除from_idx之后的所有消息
            conv.messages = conv.messages[:from_idx + 1]
            conv.updated_at = time.time()
            conv.kv_cache_key = None  # 清除缓存关键字
            
            return conv.messages
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            成功返回True
        """
        with self._lock:
            if session_id not in self.conversations:
                return False
            
            del self.conversations[session_id]
            
            # 如果删除的是活跃会话，切换到其他的
            if self.active_session_id == session_id:
                remaining = list(self.conversations.keys())
                self.active_session_id = remaining[0] if remaining else None
            
            return True
    
    def export_session(self, session_id: str) -> Optional[str]:
        """
        导出会话为JSON格式
        
        Args:
            session_id: 会话ID
            
        Returns:
            JSON字符串，如果会话不存在返回None
        """
        with self._lock:
            conv = self.conversations.get(session_id)
            if not conv:
                return None
            
            return json.dumps(conv.to_dict(), indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取会话管理器的统计信息"""
        with self._lock:
            total_messages = sum(len(conv.messages) for conv in self.conversations.values())
            total_tokens = sum(
                len(msg.token_ids) 
                for conv in self.conversations.values() 
                for msg in conv.messages
            )
            
            return {
                "total_sessions": len(self.conversations),
                "active_session_id": self.active_session_id,
                "total_messages": total_messages,
                "total_tokens": total_tokens,
                "max_sessions": self.max_sessions,
            }
