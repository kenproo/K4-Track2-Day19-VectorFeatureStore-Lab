"""Demo script running 5 queries with HybridMemoryAgent.

Deliverable for Day 19 Bonus Challenge.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent


def main() -> int:
    print("==================================================")
    print("Day 19 Bonus Challenge — Hybrid Memory Agent Demo")
    print("Combining Episodic Memory (Qdrant) + User Profile (Feast)")
    print("==================================================\n")

    agent = HybridMemoryAgent()
    user_id = "u_001"

    # Pre-populate some episodic memories for user_id
    print("1. Ingesting user episodic memories...")
    memories = [
        "Đã hoàn thành bài đọc về kiến trúc Kubernetes Pod lifecycle và auto-scaling trên AWS EKS.",
        "Đã ghi chú về cách cấu hình bảo mật Cloud Security với Zero-Trust, TLS 1.3 và AWS IAM Roles.",
        "Đang nghiên cứu kỹ thuật RAG với hybrid search kết hợp BM25 Okapi và Qdrant vector database.",
        "Tìm hiểu về kiến trúc Microservices và cơ chế circuit breaker tránh cascading failure.",
        "Ghi chép so sánh hiệu năng giữa FastAPI và gRPC trong các hệ thống phân tán thời gian thực.",
    ]
    for mem in memories:
        agent.remember(mem, user_id=user_id)
    print(f"   Indexed {len(memories)} episodic memory items for {user_id}.\n")

    # 5 demonstration queries
    queries = [
        ("Query 1 (Vector hit)", "Tôi đã đọc gì về Kubernetes?"),
        ("Query 2 (Profile context)", "Recommend đọc gì tiếp theo?"),
        ("Query 3 (Recent activity)", "Tôi đang quan tâm gì gần đây?"),
        ("Query 4 (Paraphrase vector)", "Tài liệu về tự động mở rộng hạ tầng?"),
        ("Query 5 (Mixed episodic + profile)", "Cho tôi summary cloud security"),
    ]

    print("2. Running 5 demonstration queries:")
    print("-" * 50)
    for q_label, q_text in queries:
        print(f"\n>>> [{q_label}] '{q_text}'")
        context = agent.recall(q_text, user_id=user_id, top_k=2)
        print(context)

    print("\n" + "=" * 50)
    print("Demo completed successfully (exit 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
