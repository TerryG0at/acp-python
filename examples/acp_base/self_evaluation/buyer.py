import threading
import json
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, List

from dotenv import load_dotenv

from virtuals_acp import ACPMemo
from virtuals_acp.client import VirtualsACP
from virtuals_acp.env import EnvSettings
from virtuals_acp.job import ACPJob
from virtuals_acp.models import (
    ACPAgentSort,
    ACPJobPhase,
    ACPGraduationStatus,
    ACPOnlineStatus,
)

load_dotenv(override=True)

# --------------------------
# Settings you might adjust
# --------------------------
SELLER_NAME = "BitSight"     # the provider agent name (or keyword)
OFFERING_INDEX = 0            # pick the first offering on that agent


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
            with job_queue_lock:
                job_queue.append((job, memo_to_sign))
        else:
            job_queue.append((job, memo_to_sign))

    def safe_pop_job():
        if use_thread_lock:
            with job_queue_lock:
                if job_queue:
                    return job_queue.popleft()
        else:
            if job_queue:
                return job_queue.popleft()
        return None, None

    # -------- callbacks / handlers --------
    def on_new_task(job: ACPJob, memo_to_sign: Optional[ACPMemo] = None):
        print(f"[on_new_task] Received job {job.id} (phase: {job.phase})")
        safe_append_job(job, memo_to_sign)
        job_event.set()

    def extract_text_deliverable(job) -> str:
        """Pull a text deliverable from job memos, if present."""
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
        md = extract_text_deliverable(job)
        if md:
            print("\n===== DELIVERABLE (Prediction) =====\n")
            print(md)
            print("\n====================================\n")
        else:
            print("No text deliverable found on this job.")

        # accept completion
        for memo in job.memos:
            if memo.next_phase == ACPJobPhase.COMPLETED:
                job.evaluate(True)
                break

    def process_job(job, memo_to_sign=None):
        # When provider requests payment, pay automatically
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
                    print(f"❌ Error processing job: {e}")
            if use_thread_lock:
                with job_queue_lock:
                    if not job_queue:
                        job_event.clear()
            else:
                if not job_queue:
                    job_event.clear()

    threading.Thread(target=job_worker, daemon=True).start()

    # -------- create ACP client --------
    acp = VirtualsACP(
        wallet_private_key=env.WHITELISTED_WALLET_PRIVATE_KEY,
        agent_wallet_address=env.BUYER_AGENT_WALLET_ADDRESS,
        on_new_task=on_new_task,
        on_evaluate=on_evaluate,
        entity_id=env.BUYER_ENTITY_ID,
    )

    # -------- find seller / offering --------
    agents = acp.browse_agents(
        keyword=SELLER_NAME,
        cluster=None,
        sort_by=[ACPAgentSort.SUCCESSFUL_JOB_COUNT],
        top_k=5,
        graduation_status=ACPGraduationStatus.ALL,
        online_status=ACPOnlineStatus.ALL,
    )
    if not agents:
        print("No agents found.")
        return
    seller = agents[0]
    offering = seller.offerings[OFFERING_INDEX]

    print("\nEnter requests like:")
    print("  prices=30000 horizonDays=1")
    print("Press ENTER to re-use last prices; type 'q' to quit.\n")

    last_prices = None
    while True:
        raw = input("Request> ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            break

        # parse k=v pairs
        data = dict(x.split("=", 1) for x in raw.split() if "=" in x)

        # prices
        prices_str = data.get("prices")
        if prices_str:
            try:
                prices = [float(p) for p in prices_str.split(",") if p.strip()]
                last_prices = prices  # remember
            except Exception:
                print("Could not parse prices. Use comma-separated numbers.")
                continue
        else:
            if last_prices is None:
                print("Please provide prices on first request.")
                continue
            prices = last_prices

        # horizonDays (optional, default 1)
        try:
            horizon = int(data.get("horizonDays", "1"))
        except Exception:
            horizon = 1
        horizon = max(1, min(horizon, 7))

        service_requirement = {
            "prices": prices,
            "horizonDays": horizon,
        }

        with initiate_job_lock:
            job_id = offering.initiate_job(
                service_requirement=service_requirement,
                evaluator_address=env.BUYER_AGENT_WALLET_ADDRESS,
                expired_at=datetime.now() + timedelta(hours=2),
            )
            print(
                f"Job {job_id} initiated. Waiting for negotiation/payment/evaluation...\n"
            )

    # keep listening for events
    threading.Event().wait()


if __name__ == "__main__":
    buyer()
