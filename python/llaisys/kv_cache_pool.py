"""
KV-Cache池 - 支持前缀匹配和缓存管理
实现高效的KV-Cache复用机制，支持修改历史消息时最大化缓存利用
"""

import hashlib
import time
import threading
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """缓存条目"""
    cache_key: str               # 缓存的唯一标识符
    token_ids_hash: str          # token序列的哈希值（前缀）
    token_ids_list: List[int]    # 完整的token列表
    kv_state: Any                # KV cache状态（来自llaisys）
    
    session_id: str              # 所属会话ID
    access_count: int = 1        # 访问次数
    created_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    
    def touch(self):
        """更新访问时间和计数"""
        self.last_access_time = time.time()
        self.access_count += 1


class KVCachePool:
    """
    KV-Cache池 - 管理推理中间结果的缓存
    
    特点：
    1. 支持前缀匹配：当输入token序列改变时，找最长匹配前缀
    2. LRU策略：自动清理低频访问的缓存
    3. 会话隔离：每个会话的缓存独立管理
    4. 线程安全：使用锁保护共享数据结构
    """
    
    def __init__(self, max_cache_entries: int = 100, max_memory_mb: float = 512.0):
        """
        初始化KV-Cache池
        
        Args:
            max_cache_entries: 最大缓存条目数
            max_memory_mb: 最大内存占用（MB，仅用于统计）
        """
        self._cache_entries: List[CacheEntry] = []
        self.max_cache_entries = max_cache_entries
        self.max_memory_mb = max_memory_mb
        self._lock = threading.RLock()
        
        # 统计信息
        self.total_hits = 0
        self.total_misses = 0
        self.total_matches = 0
    
    # ========================================================================
    # 哈希计算 - 用于高效的前缀匹配
    # ========================================================================
    
    @staticmethod
    def compute_hash(token_ids: List[int]) -> str:
        """
        计算token序列的哈希值
        使用SHA256而不是rolling hash，以获得更强的冲突抵抗性
        
        Args:
            token_ids: token ID列表
            
        Returns:
            哈希值（十六进制字符串）
        """
        if not token_ids:
            return ""
        
        # 将token IDs序列化为字节
        token_bytes = bytes(token_ids)
        return hashlib.sha256(token_bytes).hexdigest()
    
    @staticmethod
    def compute_hash_incremental(prefix_hash: str, token_ids: List[int], new_tokens: List[int]) -> str:
        """
        增量计算哈希值（从前缀哈希扩展）
        为了简化，这里直接重新计算，而不是真正的增量计算
        
        Args:
            prefix_hash: 前缀的哈希值
            token_ids: 原token列表
            new_tokens: 新增的tokens
            
        Returns:
            新的哈希值
        """
        combined = token_ids + new_tokens
        return KVCachePool.compute_hash(combined)
    
    # ========================================================================
    # 前缀匹配 - 核心功能
    # ========================================================================
    
    def find_longest_match(self, token_ids: List[int], session_id: str) -> Optional[Tuple[CacheEntry, int]]:
        """
        在缓存中查找最长的前缀匹配
        
        这是KV-Cache复用的关键：当用户修改历史消息时，我们找到最长的匹配前缀，
        然后只需要从该位置继续推理，而不是重新推理整个序列。
        
        复杂度：O(n*m) 其中n是缓存条数，m是token_ids长度
        可通过B-tree或Trie结构优化，但对于通常的conversation大小足够
        
        Args:
            token_ids: 当前的token ID列表
            session_id: 会话ID
            
        Returns:
            (缓存条目, 匹配长度) 或 None
        """
        with self._lock:
            if not token_ids:
                return None
            
            best_match: Optional[Tuple[CacheEntry, int]] = None
            best_match_len = 0
            
            for entry in self._cache_entries:
                # 只在同一会话内查找
                if entry.session_id != session_id:
                    continue
                
                # 计算最长公共前缀长度
                match_len = 0
                entry_tokens = entry.token_ids_list
                
                for i in range(min(len(token_ids), len(entry_tokens))):
                    if token_ids[i] == entry_tokens[i]:
                        match_len = i + 1
                    else:
                        break
                
                # 更新最佳匹配
                if match_len > best_match_len:
                    best_match_len = match_len
                    best_match = (entry, match_len)
            
            if best_match:
                self.total_matches += 1
                best_match[0].touch()
                return best_match
            
            self.total_misses += 1
            return None
    
    # ========================================================================
    # 缓存存储与检索
    # ========================================================================
    
    def store_cache(self, token_ids: List[int], kv_state: Any, 
                   session_id: str, cache_key: Optional[str] = None) -> str:
        """
        存储KV-Cache到池中
        
        Args:
            token_ids: token ID序列
            kv_state: KV cache状态对象
            session_id: 所属会话ID
            cache_key: 自定义的缓存键（如果不提供则自动生成）
            
        Returns:
            缓存键
        """
        with self._lock:
            # 生成或使用提供的缓存键
            if not cache_key:
                cache_key = f"{session_id}_{len(self._cache_entries)}_{int(time.time() * 1000)}"
            
            # 计算哈希
            token_hash = self.compute_hash(token_ids)
            
            # 创建缓存条目
            entry = CacheEntry(
                cache_key=cache_key,
                token_ids_hash=token_hash,
                token_ids_list=token_ids.copy(),
                kv_state=kv_state,
                session_id=session_id,
            )
            
            # 检查是否需要淘汰
            self._maybe_evict()
            
            # 添加到缓存列表
            self._cache_entries.append(entry)
            
            return cache_key
    
    def get_cache(self, cache_key: str) -> Optional[Any]:
        """
        获取指定缓存键的KV-Cache状态
        
        Args:
            cache_key: 缓存键
            
        Returns:
            KV cache状态，如果不存在返回None
        """
        with self._lock:
            for entry in self._cache_entries:
                if entry.cache_key == cache_key:
                    entry.touch()
                    self.total_hits += 1
                    return entry.kv_state
            
            self.total_misses += 1
            return None
    
    def get_cache_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """获取完整的缓存条目"""
        with self._lock:
            for entry in self._cache_entries:
                if entry.cache_key == cache_key:
                    entry.touch()
                    return entry
            return None
    
    # ========================================================================
    # 缓存失效
    # ========================================================================
    
    def invalidate_cache(self, cache_key: str) -> bool:
        """
        删除指定的缓存条目
        
        Args:
            cache_key: 缓存键
            
        Returns:
            成功返回True
        """
        with self._lock:
            for i, entry in enumerate(self._cache_entries):
                if entry.cache_key == cache_key:
                    del self._cache_entries[i]
                    return True
            return False
    
    def invalidate_session_cache(self, session_id: str) -> int:
        """
        删除指定会话的所有缓存
        
        Args:
            session_id: 会话ID
            
        Returns:
            删除的缓存条数
        """
        with self._lock:
            initial_count = len(self._cache_entries)
            self._cache_entries = [e for e in self._cache_entries if e.session_id != session_id]
            return initial_count - len(self._cache_entries)
    
    def clear_all(self) -> int:
        """清空所有缓存"""
        with self._lock:
            count = len(self._cache_entries)
            self._cache_entries = []
            return count
    
    # ========================================================================
    # 内存管理
    # ========================================================================
    
    def _maybe_evict(self):
        """
        当缓存满时执行淘汰策略 - LRU + 访问频率混合
        
        淘汰策略：
        1. 如果缓存条数超过max_cache_entries
        2. 计算每个条目的分数：score = access_count / (current_time - last_access_time + 1)
        3. 淘汰分数最低的条目
        """
        if len(self._cache_entries) < self.max_cache_entries:
            return
        
        # 需要淘汰 10% 的低分条目
        target_count = int(self.max_cache_entries * 0.9)
        num_to_remove = len(self._cache_entries) - target_count
        
        if num_to_remove <= 0:
            return
        
        current_time = time.time()
        
        # 计算分数
        entries_with_scores = []
        for entry in self._cache_entries:
            # 分数 = 访问次数 / 时间衰减因子
            time_decay = (current_time - entry.last_access_time) / 60.0 + 1.0  # 分钟级衰减
            score = entry.access_count / time_decay
            entries_with_scores.append((score, entry))
        
        # 按分数排序并删除最低分的条目
        entries_with_scores.sort(key=lambda x: x[0])
        for i in range(num_to_remove):
            self._cache_entries.remove(entries_with_scores[i][1])
    
    def cleanup_expired(self, max_age_seconds: float = 3600.0) -> int:
        """
        清理过期的缓存条目（基于时间）
        
        Args:
            max_age_seconds: 最大存活时间（秒）
            
        Returns:
            清理的条目数
        """
        with self._lock:
            current_time = time.time()
            initial_count = len(self._cache_entries)
            
            self._cache_entries = [
                e for e in self._cache_entries
                if (current_time - e.created_time) < max_age_seconds
            ]
            
            return initial_count - len(self._cache_entries)
    
    # ========================================================================
    # 统计和调试
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓存池统计信息"""
        with self._lock:
            if self.total_hits + self.total_misses == 0:
                hit_rate = 0.0
            else:
                hit_rate = self.total_hits / (self.total_hits + self.total_misses)
            
            if self.total_matches == 0:
                match_rate = 0.0
            else:
                match_rate = self.total_matches / (self.total_hits + self.total_misses)
            
            return {
                "total_entries": len(self._cache_entries),
                "max_entries": self.max_cache_entries,
                "total_hits": self.total_hits,
                "total_misses": self.total_misses,
                "total_matches": self.total_matches,
                "hit_rate": f"{hit_rate * 100:.2f}%",
                "match_rate": f"{match_rate * 100:.2f}%",
                "entries_by_session": self._get_entries_by_session(),
            }
    
    def _get_entries_by_session(self) -> Dict[str, int]:
        """统计每个会话的缓存条数"""
        result: Dict[str, int] = {}
        for entry in self._cache_entries:
            if entry.session_id not in result:
                result[entry.session_id] = 0
            result[entry.session_id] += 1
        return result
    
    def debug_list_entries(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出所有缓存条目（用于调试）
        
        Args:
            session_id: 过滤指定会话（可选）
            
        Returns:
            条目信息列表
        """
        with self._lock:
            result = []
            for entry in self._cache_entries:
                if session_id and entry.session_id != session_id:
                    continue
                
                result.append({
                    "cache_key": entry.cache_key,
                    "session_id": entry.session_id,
                    "token_count": len(entry.token_ids_list),
                    "token_ids_hash": entry.token_ids_hash[:16] + "...",
                    "access_count": entry.access_count,
                    "created_time": entry.created_time,
                    "last_access_time": entry.last_access_time,
                    "age_seconds": time.time() - entry.created_time,
                })
            
            return sorted(result, key=lambda x: x["last_access_time"], reverse=True)
