"""测试脚本：对 validation.jsonl 的 10 道题逐一测试"""
import asyncio
import json
import logging
import os
import sys
import time

# 项目根目录（scripts/ 的上一级）
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将项目根目录加入 sys.path，以便 import 项目模块
sys.path.insert(0, _ROOT_DIR)

# 加载 .env
from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

from config import setup_logging
from agent_loop import run_agent

_LOG_PATH = os.path.join(_ROOT_DIR, "log_validation.txt")
setup_logging(log_file=_LOG_PATH)

logger = logging.getLogger(__name__)


async def main():
    val_path = os.path.join(_ROOT_DIR, "sample", "validation.jsonl")
    with open(val_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    # 如果命令行指定了题号，只跑那道题
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        items = [items[idx]]

    correct = 0
    total = len(items)

    for i, item in enumerate(items):
        question = item["question"]
        expected = item["answer"]

        logger.info(f"{'='*60}")
        logger.info(f"题目 {i+1}/{total}")
        logger.info(f"问题: {question[:80]}...")
        logger.info(f"期望答案: {expected}")
        logger.info("-" * 60)

        start = time.time()
        try:
            answer = await run_agent(question)
        except Exception as e:
            answer = f"[ERROR] {e}"
        elapsed = time.time() - start

        # 精确匹配（归一化后比对，与赛题评测规则一致）
        exp_norm = expected.strip().rstrip(".").lower()
        ans_norm = answer.strip().rstrip(".").lower()
        is_correct = ans_norm == exp_norm

        if is_correct:
            correct += 1
            status = "CORRECT"
        else:
            status = "WRONG"

        logger.info(f"Agent答案: {answer}")
        logger.info(f"结果: {status}  耗时: {elapsed:.1f}s")

    logger.info(f"{'='*60}")
    logger.info(f"总计: {correct}/{total} 正确 ({correct/total*100:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
