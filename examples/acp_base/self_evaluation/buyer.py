import threading
import json
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

from virtuals_acp import ACPMemo
from virtuals_acp.client import VirtualsACP
from virtuals_acp.env import EnvSettings
from virtuals_acp.job import ACPJob
from virtuals_acp.models import ACPAgentSort, ACPJobPhase, ACPGraduationStatus, ACPOnlineStatus

load_dotenv(override=True)


def buyer(use_thread_lock: bool = True):
    env = EnvSettings()

    if env.WHITELISTED_WALLET_PRIVATE_KEY is None:
        raise ValueError("WHITELISTED_WALLET_PRIVATE_KEY is not set")
    if env.BUYER_AGENT_WALLET_ADDRESS is None:
        raise ValueError("BUYER_AGENT_WALLET_ADDRESS is not set")
    if env.BUYER_ENTITY_ID is None:
        raise ValueError("BUYER_ENTITY_ID is not set")

    job_queue = deque()
    job_queue_lock = threading.Lock()
    initiate_job_lock = threading.Lock()
    job_event = threading.Event()

    def safe_append_job(job, memo_to_sign: Optional[ACPMemo] = None):
        if use_thread_lock:
            print(f"[safe_append_job] Acquiring lock to append job {job.id}")
            with job_queue_lock:
                print(f"[safe_append_job] Lock acquired, appending job {job.id} to queue")
                job_queue.append((job, memo_to_sign))
        else:
            job_queue.append((job, memo_to_sign))

    def safe_pop_job():
        """Modified to return both job and memo_to_sign"""
        if use_thread_lock:
            print(f"[safe_pop_job] Acquiring lock to pop job")
            with job_queue_lock:
                if job_queue:
                    job, memo_to_sign = job_queue.popleft()
                    print(f"[safe_pop_job] Lock acquired, popped job {job.id}")
                    return job, memo_to_sign
                else:
                    print("[safe_pop_job] Queue is empty after acquiring lock")
        else:
            if job_queue:
                job, memo_to_sign = job_queue.popleft()
                print(f"[safe_pop_job] Popped job {job.id} without lock")
                return job, memo_to_sign
            else:
                print("[safe_pop_job] Queue is empty (no lock)")
        return None, None

    def job_worker():
        while True:
            job_event.wait()
            while True:
                job, memo_to_sign = safe_pop_job()
                if not job:
                    break
                try:
                    process_job(job, memo_to_sign)
                except Exception as e:
                    print(f"\u274c Error processing job: {e}")
            if use_thread_lock:
                with job_queue_lock:
                    if not job_queue:
                        job_event.clear()
            else:
                if not job_queue:
                    job_event.clear()

    def on_new_task(job: ACPJob, memo_to_sign: Optional[ACPMemo] = None):
        print(f"[on_new_task] Received job {job.id} (phase: {job.phase})")
        safe_append_job(job, memo_to_sign)
        job_event.set()

    def extract_text_deliverable(job) -> str:
        for m in job.memos or []:
            content = getattr(m, "content", None)
            if isinstance(content, dict) and content.get("type") == "text":
                return str(content.get("value", ""))
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                    if data.get("type") == "text":
                        return str(data.get("value", ""))
                except Exception:
                    pass
        return ""

    def on_evaluate(job: ACPJob):
        quiz_md = extract_text_deliverable(job)
        if quiz_md:
            print("\n===== DELIVERABLE (MCQ) =====\n")
            print(quiz_md)
            print("\n=============================\n")
        else:
            print("No text deliverable found on this job.")

        # accept completion
        for memo in job.memos:
            if memo.next_phase == ACPJobPhase.COMPLETED:
                job.evaluate(True)
                break

    def process_job(job, memo_to_sign=None):
        if job.phase == ACPJobPhase.NEGOTIATION:
            for memo in job.memos:
                if memo.next_phase == ACPJobPhase.TRANSACTION:
                    print("Paying job", job.id)
                    job.pay(job.price)
                    break
        elif job.phase == ACPJobPhase.COMPLETED:
            print("Job completed", job)
        elif job.phase == ACPJobPhase.REJECTED:
            print("Job rejected", job)

    threading.Thread(target=job_worker, daemon=True).start()

    acp = VirtualsACP(
        wallet_private_key=env.WHITELISTED_WALLET_PRIVATE_KEY,
        agent_wallet_address=env.BUYER_AGENT_WALLET_ADDRESS,
        on_new_task=on_new_task,
        on_evaluate=on_evaluate,
        entity_id=env.BUYER_ENTITY_ID
    )

    # find your seller (by name or keyword)
    agents = acp.browse_agents(
        keyword="NerdyLexy",          # or any keyword/tag you set
        sort_by=[ACPAgentSort.SUCCESSFUL_JOB_COUNT],
        top_k=5,
        graduation_status=ACPGraduationStatus.ALL,
        online_status=ACPOnlineStatus.ALL
    )
    if not agents:
        print("No agents found.")
        return
    seller = agents[0]
    offering = seller.offerings[0]    # pick “MCQ” offering

    print("\nType requests like:  topic=Fish  num=5")
    print("Press ENTER to use defaults; type 'q' to quit.\n")

    while True:
        raw = input("Request> ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break

        # Parse k=v pairs typed in terminal
        data = dict(x.split("=", 1) for x in raw.split() if "=" in x)
        topic = data.get("topic", "general knowledge")
        num = int(data.get("num", "5"))
        text = data.get("text", "")  # optional notes

        # Send exactly these to the seller
        service_requirement = {
            "topic": topic,
            "num": num,
            "text": text
        }

        job_id = offering.initiate_job(
            service_requirement=service_requirement,
            evaluator_address=env.BUYER_AGENT_WALLET_ADDRESS,
            expired_at=datetime.now() + timedelta(hours=2),
        )
        print(f"Job {job_id} initiated. Waiting for negotiation/payment/evaluation...\n")

    # keep listening
    threading.Event().wait()

if __name__ == "__main__":
    buyer()
