"""全量测试脚本：对 question.jsonl 的 100 道题逐一测试，输出符合赛题要求的 JSONL 结果文件"""
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

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

from config import setup_logging
from agent_loop import run_agent

_LOG_PATH = os.path.join(_ROOT_DIR, "log.txt")
setup_logging(log_file=_LOG_PATH)

logger = logging.getLogger(__name__)

# 结果输出路径
OUTPUT_PATH = os.path.join(_ROOT_DIR, "sample", "result.jsonl")
# 进度记录路径（用于断点续跑）
PROGRESS_PATH = os.path.join(_ROOT_DIR, "sample", "progress.json")


def load_progress():
    """加载已完成的结果，支持断点续跑"""
    done = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    done[item["id"]] = item["answer"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def save_result(qid: int, answer: str):
    """追加写入一条结果"""
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": qid, "answer": answer}, ensure_ascii=False) + "\n")


def rebuild_output(results: dict):
    """按 id 排序重写完整结果文件"""
    sorted_items = sorted(results.items(), key=lambda x: x[0])
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for qid, answer in sorted_items:
            f.write(json.dumps({"id": qid, "answer": answer}, ensure_ascii=False) + "\n")


async def run_single(qid: int, question: str, index: int, total: int) -> str:
    """运行单道题，返回答案"""
    logger.info(f"{'='*60}")
    logger.info(f"[{index+1}/{total}] 题目 ID: {qid}")
    logger.info(f"问题: {question[:100]}{'...' if len(question) > 100 else ''}")
    logger.info("-" * 60)

    start = time.time()
    try:
        answer = await run_agent(question)
    except Exception as e:
        answer = f"ERROR"
        logger.info(f"  [异常] {e}")
    elapsed = time.time() - start

    logger.info(f"  答案: {answer}")
    logger.info(f"  耗时: {elapsed:.1f}s")
    return answer


async def main():
    question_path = os.path.join(_ROOT_DIR, "sample", "question.jsonl")
    with open(question_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    # 支持命令行参数
    # 用法:
    #   python scripts/test_full.py              -- 跑全部 100 题（支持断点续跑）
    #   python scripts/test_full.py 5            -- 只跑 id=5 的题
    #   python scripts/test_full.py 0-9          -- 跑 id 0~9 的题
    #   python scripts/test_full.py --clean      -- 清除进度，从头开始
    #   python scripts/test_full.py --rerun 3,7  -- 重跑指定 id 的题

    # 解析参数
    clean = False
    rerun_ids = None
    filter_ids = None

    for arg in sys.argv[1:]:
        if arg == "--clean":
            clean = True
        elif arg.startswith("--rerun"):
            pass
        elif "--rerun" in sys.argv and arg != "--rerun":
            rerun_ids = [int(x) for x in arg.split(",")]
        elif "-" in arg and not arg.startswith("-"):
            parts = arg.split("-")
            filter_ids = list(range(int(parts[0]), int(parts[1]) + 1))
        elif arg.isdigit():
            filter_ids = [int(arg)]

    # 清除进度
    if clean:
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)
        logger.info("已清除历史结果，从头开始。")

    # 加载已完成结果
    done = {} if clean else load_progress()

    # 处理重跑：从已完成中移除指定 id
    if rerun_ids:
        for rid in rerun_ids:
            done.pop(rid, None)
        # 重写结果文件（去掉要重跑的）
        rebuild_output(done)
        filter_ids = rerun_ids

    # 筛选要跑的题目
    if filter_ids is not None:
        items = [it for it in items if it["id"] in filter_ids]

    # 过滤掉已完成的
    pending = [it for it in items if it["id"] not in done]

    if not pending:
        logger.info(f"所有 {len(items)} 道题已完成，结果文件: {OUTPUT_PATH}")
        logger.info("如需重跑，使用: python scripts/test_full.py --clean 或 python scripts/test_full.py --rerun 0,1,2")
        return

    total_all = len(items)
    total_pending = len(pending)
    logger.info(f"总题数: {total_all}, 已完成: {total_all - total_pending}, 待运行: {total_pending}")
    logger.info(f"结果输出: {OUTPUT_PATH}")

    start_all = time.time()

    for i, item in enumerate(pending):
        qid = item["id"]
        question = item["question"]

        answer = await run_single(qid, question, i, total_pending)
        save_result(qid, answer)
        done[qid] = answer

    elapsed_all = time.time() - start_all

    # 最终按 id 排序重写结果文件
    rebuild_output(done)

    logger.info(f"{'='*60}")
    logger.info(f"全部完成！共 {len(done)} 道题")
    logger.info(f"总耗时: {elapsed_all:.1f}s")
    logger.info(f"结果文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
