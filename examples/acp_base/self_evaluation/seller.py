import os
import json
import threading
import time
from collections import deque
from typing import Optional, List

# numeric / model deps
import numpy as np
try:
    import joblib  # pip install joblib
except ImportError:
    joblib = None

from dotenv import load_dotenv

from virtuals_acp import VirtualsACP, ACPJob, ACPJobPhase, ACPMemo, IDeliverable
from virtuals_acp.env import EnvSettings

load_dotenv(override=True)

# ----------------------------
# Model artifacts (YOUR files)
# ----------------------------
MODEL_PATH  = os.path.join("model", "bitcoin_price_model.pkl")
SCALER_PATH = os.path.join("model", "bitcoin_price_scaler.pkl")

MODEL = None
SCALER = None

def _load_artifacts():
    global MODEL, SCALER
    if joblib is None:
        print("joblib not installed. Run: pip install joblib")
        return
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
            print(f"Loaded model: {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}")

    if os.path.exists(SCALER_PATH):
        try:
            SCALER = joblib.load(SCALER_PATH)
            print(f"Loaded scaler: {SCALER_PATH}")
        except Exception as e:
            print(f"Failed to load scaler: {e}")
    else:
        print("Scaler not provided (optional).")

def prepare_features(prices: List[float]) -> np.ndarray:
    """
    Build one feature row from the last N prices where N is
    the model's n_features_in_ if available, else 30.
    Applies the scaler if provided.
    """
    if MODEL is None:
        raise RuntimeError("MODEL not loaded")

    if hasattr(MODEL, "n_features_in_"):
        window = int(MODEL.n_features_in_)
    else:
        window = 30  # fallback if model doesn't expose it

    if len(prices) < window:
        raise ValueError(f"Need at least {window} prices, got {len(prices)}")

    x = np.array(prices[-window:], dtype=float)
    if SCALER is not None:
        x = SCALER.transform(x.reshape(-1, 1)).ravel()

    return x.reshape(1, -1)

def predict_next(prices: List[float], horizon_days: int = 1) -> float:
    """
    Predict next-day BTC price with your trained model.
    If anything fails, fall back to a simple moving average.
    """
    try:
        X = prepare_features(prices)
        y_hat = MODEL.predict(X)
        return float(np.asarray(y_hat).ravel()[0])
    except Exception as e:
        print(f"[predict_next] Falling back due to: {e}")
        k = min(5, len(prices))
        return float(sum(prices[-k:]) / k)

# -------------------------------------------------
# ACP seller: same structure, different deliverable
# -------------------------------------------------
def seller(use_thread_lock: bool = True):
    env = EnvSettings()

    if env.WHITELISTED_WALLET_PRIVATE_KEY is None:
        raise ValueError("WHITELISTED_WALLET_PRIVATE_KEY is not set")
    if env.SELLER_AGENT_WALLET_ADDRESS is None:
        raise ValueError("SELLER_AGENT_WALLET_ADDRESS is not set")
    if env.SELLER_ENTITY_ID is None:
        raise ValueError("SELLER_ENTITY_ID is not set")

    # load ML artifacts once
    _load_artifacts()

    job_queue = deque()
    job_queue_lock = threading.Lock()
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

    def job_worker():
        while True:
            job_event.wait()
            while True:
                job, memo_to_sign = safe_pop_job()
                if not job:
                    break
                threading.Thread(
                    target=handle_job_with_delay,
                    args=(job, memo_to_sign),
                    daemon=True
                ).start()
            if use_thread_lock:
                with job_queue_lock:
                    if not job_queue:
                        job_event.clear()
            else:
                if not job_queue:
                    job_event.clear()

    def handle_job_with_delay(job, memo_to_sign):
        try:
            process_job(job, memo_to_sign)
            time.sleep(1.5)
        except Exception as e:
            print(f"❌ Error processing job: {e}")

    def on_new_task(job: ACPJob, memo_to_sign: Optional[ACPMemo] = None):
        print(f"[on_new_task] job {job.id} (phase: {job.phase})")
        safe_append_job(job, memo_to_sign)
        job_event.set()

    def extract_requirement_from_memos(job) -> dict:
        """Return the first serviceRequirement found in job memos."""
        for m in job.memos or []:
            content = getattr(m, "content", None)
            if isinstance(content, dict):
                sr = content.get("serviceRequirement")
                if sr is not None:
                    return sr
            elif isinstance(content, str):
                try:
                    data = json.loads(content)
                    sr = data.get("serviceRequirement")
                    if sr is not None:
                        return sr
                except Exception:
                    pass
        return {}

    def process_job(job: ACPJob, memo_to_sign: Optional[ACPMemo] = None):
        if (
            job.phase == ACPJobPhase.REQUEST
            and memo_to_sign is not None
            and memo_to_sign.next_phase == ACPJobPhase.NEGOTIATION
        ):
            job.respond(True)

        elif (
            job.phase == ACPJobPhase.TRANSACTION
            and memo_to_sign is not None
            and memo_to_sign.next_phase == ACPJobPhase.EVALUATION
        ):
            # ---- Prediction service here ----
            req = extract_requirement_from_memos(job)

            prices = req.get("prices", [])
            horizon = int(req.get("horizonDays", 1))

            # basic validation
            if not isinstance(prices, list) or not all(isinstance(x, (int, float)) for x in prices):
                job.deliver(IDeliverable(
                    type="text",
                    value="Invalid 'prices': supply an array of numbers (oldest → newest)."
                ))
                return

            horizon = max(1, min(horizon, 7))

            pred = predict_next(prices, horizon)

            md = (
                "## BTC Price Prediction\n\n"
                f"- Horizon: **{horizon} day(s)**\n"
                f"- Last known price: **${prices[-1]:,.2f}**\n"
                f"- Predicted price in {horizon} day(s): **${pred:,.2f}**\n"
            )
            job.deliver(IDeliverable(type="text", value=md))

        elif job.phase == ACPJobPhase.COMPLETED:
            print(f"Job completed {job.id}")
        elif job.phase == ACPJobPhase.REJECTED:
            print(f"Job rejected {job.id}")

    threading.Thread(target=job_worker, daemon=True).start()

    VirtualsACP(
        wallet_private_key=env.WHITELISTED_WALLET_PRIVATE_KEY,
        agent_wallet_address=env.SELLER_AGENT_WALLET_ADDRESS,
        on_new_task=on_new_task,
        entity_id=env.SELLER_ENTITY_ID,
    )

    print("Waiting for new task...")
    threading.Event().wait()

if __name__ == "__main__":
    seller()
