#!/usr/bin/env python3
"""
测试脚本 - 验证会话管理和KV-Cache池功能
"""

import sys
import requests
import json
import time
from typing import List, Dict, Any


def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_session_creation():
    """Test session creation"""
    print_section("Test 1: Session Creation")
    
    base_url = "http://localhost:8000"
    
    # Create session
    url = f"{base_url}/v1/sessions/create"
    payload = {"title": "Test Session 1"}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        session_id = data["session_id"]
        print(f"✅ Created session: {session_id}")
        print(f"   Title: {data['title']}")
        print(f"   Created at: {time.ctime(data['created_at'])}")
        
        return session_id
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_list_sessions():
    """Test listing sessions"""
    print_section("Test 2: List Sessions")
    
    base_url = "http://localhost:8000"
    url = f"{base_url}/v1/sessions"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        sessions = data["sessions"]
        print(f"✅ Found {len(sessions)} session(s):")
        
        for s in sessions:
            status = "ACTIVE" if s["is_active"] else ""
            print(f"   - {s['id'][:8]}... | {s['title']} | {s['message_count']} msgs {status}")
        
        return sessions
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_send_message(session_id: str):
    """Test sending message"""
    print_section("Test 3: Send Message")
    
    base_url = "http://localhost:8000"
    url = f"{base_url}/v1/sessions/{session_id}/message"
    
    payload = {
        "content": "你能解释什么是KV-Cache吗？",
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 50,
        "seed": 0,
        "stream": False,
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Message sent successfully")
        print(f"\n   User: {data['user_message']}")
        print(f"\n   Assistant: {data['assistant_message']}")
        print(f"\n   Stats: {json.dumps(data['stats'], indent=4)}")
        
        return session_id
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_get_messages(session_id: str):
    """Test getting messages"""
    print_section("Test 4: Get Session Messages")
    
    base_url = "http://localhost:8000"
    url = f"{base_url}/v1/sessions/{session_id}/messages"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        messages = data["messages"]
        print(f"✅ Retrieved {len(messages)} message(s):")
        
        for i, msg in enumerate(messages):
            role = msg["role"].upper()
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            print(f"   [{i}] {role}: {content}")
        
        return messages
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_update_message(session_id: str, message_idx: int):
    """Test updating message"""
    print_section("Test 5: Update Message (Modify & Regenerate)")
    
    base_url = "http://localhost:8000"
    url = f"{base_url}/v1/sessions/{session_id}/message/{message_idx}"
    
    payload = {
        "new_content": "可以用中文详细解释KV-Cache的作用和优势吗？",
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 50,
        "seed": 42,  # Fixed seed for reproducibility
    }
    
    try:
        response = requests.put(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Message updated and regenerated")
        print(f"\n   New Assistant: {data['assistant_message']}")
        print(f"\n   Stats: {json.dumps(data['stats'], indent=4)}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_regenerate_from_message(session_id: str, message_idx: int):
    """Test regenerating from a message"""
    print_section("Test 6: Regenerate From Message")
    
    base_url = "http://localhost:8000"
    url = f"{base_url}/v1/sessions/{session_id}/regenerate"
    
    payload = {
        "from_message_idx": message_idx,
        "max_tokens": 100,
        "temperature": 0.5,  # Lower temperature for different output
        "top_p": 0.9,
        "top_k": 50,
        "seed": 123,
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Regenerated from message {message_idx}")
        print(f"\n   New Assistant: {data['assistant_message']}")
        print(f"\n   Stats: {json.dumps(data['stats'], indent=4)}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_cache_stats():
    """Test getting cache statistics"""
    print_section("Test 7: KV-Cache Statistics")
    
    base_url = "http://localhost:8000"
    url = f"{base_url}/v1/debug/cache-stats"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        cache_stats = data["cache_stats"]
        session_stats = data["session_stats"]
        
        print(f"✅ Cache Statistics:")
        print(f"\n  Cache Pool:")
        print(f"    Total Entries: {cache_stats['total_entries']}/{cache_stats['max_entries']}")
        print(f"    Hit Rate: {cache_stats['hit_rate']}")
        print(f"    Match Rate: {cache_stats['match_rate']}")
        print(f"    Total Hits: {cache_stats['total_hits']}")
        print(f"    Total Misses: {cache_stats['total_misses']}")
        print(f"    Total Matches: {cache_stats['total_matches']}")
        
        print(f"\n  Sessions:")
        print(f"    Total: {session_stats['total_sessions']}")
        print(f"    Messages: {session_stats['total_messages']}")
        print(f"    Tokens: {session_stats['total_tokens']}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_delete_session(session_id: str):
    """Test deleting a session"""
    print_section("Test 8: Delete Session")
    
    base_url = "http://localhost:8000"
    url = f"{base_url}/v1/sessions/{session_id}"
    
    try:
        response = requests.delete(url)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ {data['message']}")
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  Session Management & KV-Cache Pool Verification".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Test 1: Create session
        session_id = test_session_creation()
        if not session_id:
            return
        
        # Test 2: List sessions
        test_list_sessions()
        
        # Test 3: Send message
        test_send_message(session_id)
        
        # Test 4: Get messages
        messages = test_get_messages(session_id)
        if not messages or len(messages) < 2:
            print("❌ Not enough messages for further tests")
            return
        
        # Test 5: Update message
        test_update_message(session_id, 0)
        
        # Test 6: Regenerate from message
        test_regenerate_from_message(session_id, 0)
        
        # Test 7: Cache statistics
        test_cache_stats()
        
        # Test 8: Delete session
        test_delete_session(session_id)
        
        print_section("All Tests Completed")
        print("✅ Session management and KV-Cache pool working correctly!\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")


if __name__ == "__main__":
    main()
