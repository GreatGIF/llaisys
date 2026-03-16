"""
Enhanced LLM Chat Client with Session Management and KV-Cache Pool support
支持多会话管理、消息修改、重新生成等功能的增强客户端
"""

import requests
import json
import sys
import os
from typing import Optional, List, Dict, Any
from datetime import datetime


class LLMChatClient:
    """LLM Chat API Client with session management"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.current_session_id: Optional[str] = None
        self.sessions: Dict[str, Dict[str, Any]] = {}  # Cache session info
    
    # ========================================================================
    # Session Management
    # ========================================================================
    
    def create_session(self, title: Optional[str] = None) -> str:
        """Create a new session"""
        url = f"{self.base_url}/v1/sessions/create"
        payload = {}
        if title:
            payload["title"] = title
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        session_id = data["session_id"]
        self.sessions[session_id] = {
            "id": session_id,
            "title": data["title"],
            "created_at": data["created_at"],
            "message_count": 0,
        }
        
        if self.current_session_id is None:
            self.current_session_id = session_id
        
        return session_id
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        url = f"{self.base_url}/v1/sessions"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["sessions"]
    
    def switch_session(self, session_id: str):
        """Switch to a different session"""
        # Verify session exists
        url = f"{self.base_url}/v1/sessions/{session_id}/messages"
        response = requests.get(url)
        response.raise_for_status()
        self.current_session_id = session_id
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        url = f"{self.base_url}/v1/sessions/{session_id}"
        response = requests.delete(url)
        response.raise_for_status()
        
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        if self.current_session_id == session_id:
            self.current_session_id = None
    
    # ========================================================================
    # Message Operations
    # ========================================================================
    
    def get_messages(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all messages in a session"""
        sid = session_id or self.current_session_id
        if not sid:
            raise ValueError("No session selected")
        
        url = f"{self.base_url}/v1/sessions/{sid}/messages"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["messages"]
    
    def send_message(
        self,
        content: str,
        session_id: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 50,
        seed: int = 0,
        stream: bool = True,
    ):
        """Send a message and get response"""
        sid = session_id or self.current_session_id
        if not sid:
            raise ValueError("No session selected")
        
        url = f"{self.base_url}/v1/sessions/{sid}/message"
        payload = {
            "content": content,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "stream": stream,
        }
        
        if stream:
            return self._stream_message(url, payload)
        else:
            return self._non_stream_message(url, payload)
    
    def _non_stream_message(self, url: str, payload: Dict) -> str:
        """Non-streaming message sending"""
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        data = response.json()
        return data["assistant_message"]
    
    def _stream_message(self, url: str, payload: Dict):
        """Streaming message sending - yields token by token"""
        response = requests.post(url, json=payload, timeout=300, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line.startswith("data: "):
                data_str = line[6:]
                
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    if data.get("choices"):
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                except json.JSONDecodeError:
                    pass
    
    def update_message(
        self,
        message_idx: int,
        new_content: str,
        session_id: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 50,
        seed: int = 0,
    ) -> str:
        """Update a message and regenerate"""
        sid = session_id or self.current_session_id
        if not sid:
            raise ValueError("No session selected")
        
        url = f"{self.base_url}/v1/sessions/{sid}/message/{message_idx}"
        payload = {
            "new_content": new_content,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
        }
        
        response = requests.put(url, json=payload, timeout=300)
        response.raise_for_status()
        
        data = response.json()
        return data["assistant_message"]
    
    def regenerate_from_message(
        self,
        from_message_idx: int,
        session_id: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 50,
        seed: int = 0,
    ) -> str:
        """Regenerate response from a specific message"""
        sid = session_id or self.current_session_id
        if not sid:
            raise ValueError("No session selected")
        
        url = f"{self.base_url}/v1/sessions/{sid}/regenerate"
        payload = {
            "from_message_idx": from_message_idx,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
        }
        
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        data = response.json()
        return data["assistant_message"]
    
    # ========================================================================
    # Debugging
    # ========================================================================
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get KV-Cache pool statistics"""
        url = f"{self.base_url}/v1/debug/cache-stats"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def clear_cache(self) -> int:
        """Clear all KV-Cache"""
        url = f"{self.base_url}/v1/debug/cache-clear"
        response = requests.post(url)
        response.raise_for_status()
        return response.json()["cleared_entries"]


# ========================================================================
# Interactive CLI
# ========================================================================

class InteractiveCLI:
    """Interactive CLI for session management"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.client = LLMChatClient(base_url)
        self._cached_sessions = []  # Cache for session list (for number-based switching)
        self.available_commands = {
            "new": "创建新会话 - /new [标题]",
            "list": "列出所有会话 - /list",
            "switch": "切换会话 - /switch <序号|ID前缀|完整ID>",
            "delete": "删除会话 - /delete [session_id]",
            "messages": "显示当前会话消息 - /messages",
            "edit": "编辑消息 - /edit <msg_idx> <新内容>",
            "regen": "从消息重新生成 - /regen <msg_idx>",
            "stats": "显示缓存统计 - /stats",
            "clear-cache": "清空缓存 - /clear-cache",
            "help": "显示此帮助信息 - /help",
            "quit": "退出 - /quit",
        }
    
    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "="*70)
        print("LLM Chat Client with Session Management & KV-Cache Pool Support")
        print("="*70)
        print("类型 /help 查看命令列表")
        print("\n💡 快速提示: 使用 /list 查看所有会话，然后用 /switch 1 按序号切换")
        print("="*70 + "\n")
    
    def print_help(self):
        """Print help message"""
        print("\n" + "-"*70)
        print("可用命令:")
        for cmd, desc in self.available_commands.items():
            print(f"  {desc}")
        
        print("\n" + "💡 " + "-"*66)
        print("💡 会话切换的三种方式:")
        print("   /switch 1              - 通过序号快速切换 (最简单)")
        print("   /switch a55aedb6       - 通过ID前缀匹配")
        print("   /switch <完整ID>        - 通过完整ID切换")
        print("💡 " + "-"*66 + "\n")
    
    
    def format_message(self, msg: Dict[str, Any], idx: int) -> str:
        """Format a message for display"""
        role = msg["role"].upper()
        content = msg["content"]
        timestamp = datetime.fromtimestamp(msg.get("timestamp", 0)).strftime("%H:%M:%S")
        
        role_color = {
            "USER": "\033[94m",  # Blue
            "ASSISTANT": "\033[92m",  # Green
            "SYSTEM": "\033[93m",  # Yellow
        }.get(role, "")
        reset_color = "\033[0m"
        
        return f"[{idx}] {role_color}{role}{reset_color} ({timestamp}):\n{content}\n"
    
    def display_messages(self):
        """Display all messages in current session"""
        if not self.client.current_session_id:
            print("❌ No session selected")
            return
        
        try:
            messages = self.client.get_messages()
            if not messages:
                print("📝 No messages yet")
                return
            
            print("\n" + "="*70)
            print(f"Session: {self.client.current_session_id}")
            print("="*70)
            
            for idx, msg in enumerate(messages):
                print(self.format_message(msg, idx))
            
            # Show cache stats
            stats = self.client.get_cache_stats()
            print(f"📊 KV-Cache Hit Rate: {stats['cache_stats']['hit_rate']}")
            print()
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def display_sessions(self):
        """Display all sessions with number-based selection"""
        try:
            sessions = self.client.list_sessions()
            self._cached_sessions = sessions  # Cache for number-based switching
            
            if not sessions:
                print("📝 No sessions yet")
                return
            
            print("\n" + "="*70)
            print("Available Sessions:")
            print("="*70)
            
            for idx, session in enumerate(sessions, 1):
                is_active = "✓" if session["is_active"] else " "
                # Show: [1] ✓ title | id (full) | msg count
                session_id = session['id']
                title = session['title'][:30] + "..." if len(session['title']) > 30 else session['title']
                print(f"[{idx}] [{is_active}] {title:33} | {session_id} | {session['message_count']} msgs")
            
            print("\n💡 切换会话: /switch 1  或  /switch <完整ID>  或  /switch <ID前缀>")
            print()
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def handle_command(self, line: str):
        """Handle user command"""
        if not line.startswith("/"):
            # Regular message
            return self.send_message(line)
        
        parts = line[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == "new":
            self.cmd_new(args)
        elif cmd == "list":
            self.display_sessions()
        elif cmd == "switch":
            self.cmd_switch(args)
        elif cmd == "delete":
            self.cmd_delete(args)
        elif cmd == "messages":
            self.display_messages()
        elif cmd == "edit":
            self.cmd_edit(args)
        elif cmd == "regen":
            self.cmd_regen(args)
        elif cmd == "stats":
            self.cmd_stats()
        elif cmd == "clear-cache":
            self.cmd_clear_cache()
        elif cmd == "help":
            self.print_help()
        elif cmd == "quit":
            return False
        else:
            print(f"❌ Unknown command: /{cmd}")
        
        return True
    
    def send_message(self, content: str) -> bool:
        """Send a message"""
        if not self.client.current_session_id:
            print("❌ No session selected. Use /new to create one.")
            return True
        
        try:
            print("🤖 ", end="", flush=True)
            full_response = ""
            
            for token in self.client.send_message(content, stream=True):
                print(token, end="", flush=True)
                full_response += token
            
            print("\n")
            
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to server")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        return True
    
    def cmd_new(self, title: str):
        """Create new session"""
        try:
            title = title.strip() if title else None
            session_id = self.client.create_session(title)
            print(f"✅ Created session: {session_id[:8]}...")
            self.display_sessions()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_switch(self, session_arg: str):
        """Switch session by number, ID prefix, or full ID"""
        if not session_arg.strip():
            print("❌ Usage:")
            print("  /switch 1            (按序号)")
            print("  /switch a55aedb6     (按ID前缀)")
            print("  /switch <完整ID>      (按完整ID)")
            self.display_sessions()
            return
        
        arg = session_arg.strip()
        target_id = None
        
        # 方法1: 序号切换（最简单）
        if arg.isdigit():
            idx = int(arg)
            if 1 <= idx <= len(self._cached_sessions):
                target_id = self._cached_sessions[idx - 1]['id']
            else:
                print(f"❌ 序号 {idx} 无效，请使用 /list 查看可用会话")
                return
        
        # 方法2: ID前缀匹配
        else:
            # 从缓存会话列表中查找前缀匹配
            matches = [s for s in self._cached_sessions if s['id'].startswith(arg)]
            
            if len(matches) == 0:
                print(f"❌ 未找到匹配ID: {arg}")
                # 刷新并重新尝试
                sessions = self.client.list_sessions()
                self._cached_sessions = sessions
                matches = [s for s in sessions if s['id'].startswith(arg)]
                
                if len(matches) == 0:
                    print("❌ 仍未找到匹配的会话")
                    return
            
            if len(matches) > 1:
                print(f"⚠️  多个会话匹配此前缀，请更具体：")
                for s in matches[:5]:
                    print(f"  - {s['id']} ({s['title']})")
                return
            
            target_id = matches[0]['id']
        
        # 执行切换
        if target_id:
            try:
                self.client.switch_session(target_id)
                # 显示切换后的会话信息
                for s in self._cached_sessions:
                    if s['id'] == target_id:
                        msg_count = s['message_count']
                        title = s['title']
                        print(f"✅ 已切换到: {title} ({msg_count}条消息)")
                        print(f"   ID: {target_id}")
                        break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def cmd_delete(self, session_id: str):
        """Delete session"""
        if not session_id.strip():
            session_id = self.client.current_session_id
        
        if not session_id:
            print("❌ No session specified")
            return
        
        try:
            self.client.delete_session(session_id.strip())
            print(f"✅ Deleted session")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_edit(self, args: str):
        """Edit message and regenerate"""
        if not args.strip():
            print("❌ Usage: /edit <msg_idx> <新内容>")
            return
        
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            print("❌ Usage: /edit <msg_idx> <新内容>")
            return
        
        try:
            msg_idx = int(parts[0])
            new_content = parts[1]
            
            print("♻️  重新生成中...")
            response = self.client.update_message(msg_idx, new_content)
            print(f"🤖 {response}\n")
        
        except ValueError:
            print("❌ Invalid message index")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_regen(self, args: str):
        """Regenerate from message"""
        if not args.strip():
            print("❌ Usage: /regen <msg_idx>")
            return
        
        try:
            msg_idx = int(args.strip())
            print("♻️  重新生成中...")
            response = self.client.regenerate_from_message(msg_idx)
            print(f"🤖 {response}\n")
        
        except ValueError:
            print("❌ Invalid message index")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_stats(self):
        """Show cache statistics"""
        try:
            stats = self.client.get_cache_stats()
            
            print("\n" + "="*70)
            print("KV-Cache Pool Statistics:")
            print("="*70)
            
            cache_stats = stats["cache_stats"]
            session_stats = stats["session_stats"]
            
            print(f"Total Cache Entries: {cache_stats['total_entries']}/{cache_stats['max_entries']}")
            print(f"Hit Rate: {cache_stats['hit_rate']}")
            print(f"Match Rate: {cache_stats['match_rate']}")
            print(f"Total Hits: {cache_stats['total_hits']}")
            print(f"Total Misses: {cache_stats['total_misses']}")
            print()
            print("Sessions:")
            print(f"  Total Sessions: {session_stats['total_sessions']}")
            print(f"  Total Messages: {session_stats['total_messages']}")
            print(f"  Total Tokens: {session_stats['total_tokens']}")
            print("="*70 + "\n")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_clear_cache(self):
        """Clear KV-Cache"""
        try:
            count = self.client.clear_cache()
            print(f"✅ Cleared {count} cache entries\n")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def run(self):
        """Run interactive CLI"""
        self.print_banner()
        
        # Create initial session
        try:
            self.client.create_session("Conversation " + datetime.now().strftime("%Y-%m-%d %H:%M"))
            print(f"✅ 创建新会话\n")
        except Exception as e:
            print(f"❌ Failed to create session: {e}\n")
            return
        
        try:
            while True:
                try:
                    user_input = input("👤 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if not self.handle_command(user_input):
                        print("👋 再见!")
                        break
                
                except KeyboardInterrupt:
                    print("\n👋 再见!")
                    break
        
        except Exception as e:
            print(f"❌ Error: {e}")


# ========================================================================
# Main
# ========================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Chat Client with Session Management")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the server",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo"],
        default="interactive",
        help="Operating mode",
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        cli = InteractiveCLI(args.base_url)
        try:
            cli.run()
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to server at {args.base_url}")
            sys.exit(1)
    
    elif args.mode == "demo":
        # Demo mode - show capabilities
        client = LLMChatClient(args.base_url)
        
        try:
            print("🎬 演示：会话管理和KV-Cache池\n")
            
            # Create session
            print("1️⃣  创建会话...")
            sid1 = client.create_session("对话1 - Python编程")
            print(f"   ✅ {sid1[:8]}...\n")
            
            # Send message
            print("2️⃣  发送消息并使用KV-Cache...")
            response = ""
            for token in client.send_message("请解释Python中的装饰器", stream=True):
                response += token
            print(f"   回复长度: {len(response)} 字符\n")
            
            # Create another session
            print("3️⃣  创建第二个会话...")
            sid2 = client.create_session("对话2 - JavaScript编程")
            print(f"   ✅ {sid2[:8]}...\n")
            
            # List sessions
            print("4️⃣  列出所有会话...")
            sessions = client.list_sessions()
            for s in sessions:
                print(f"   - {s['title']} ({s['message_count']} 消息)")
            print()
            
            # Get cache stats
            print("5️⃣  缓存统计...")
            stats = client.get_cache_stats()
            print(f"   命中率: {stats['cache_stats']['hit_rate']}")
            print(f"   匹配率: {stats['cache_stats']['match_rate']}\n")
            
            print("✅ 演示完成!")
        
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to server at {args.base_url}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
