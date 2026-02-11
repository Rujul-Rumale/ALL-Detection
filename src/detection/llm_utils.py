from datetime import datetime
import time
import socket
import threading
import traceback

# Default configuration for Raspberry Pi 5
DEFAULT_MODEL = "phi3"  # Fits in 8GB RAM, reasonable speed
CONNECTION_TIMEOUT = 3.0 # Seconds
OLLAMA_PORT = 11434

try:
    import ollama
    OLLAMA_AVAILABLE = True
    print(f"[LLM] ollama library imported successfully")
except ImportError:
    OLLAMA_AVAILABLE = False
    print(f"[LLM] WARNING: ollama library NOT installed")


def _run_with_timeout(func, timeout, default=None):
    """Run a function with a hard timeout. Returns default if it hangs."""
    result = [default]
    error = [None]
    
    def wrapper():
        try:
            result[0] = func()
        except Exception as e:
            error[0] = e
    
    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    t.join(timeout=timeout)
    
    if t.is_alive():
        print(f"[LLM] TIMEOUT: {func.__name__} exceeded {timeout}s — aborting")
        return default
    if error[0]:
        print(f"[LLM] ERROR in {func.__name__}: {error[0]}")
        return default
    return result[0]


class LLMGenerator:
    """
    Generates clinical summaries based on cell metrics.
    Connects to local Ollama instance (e.g. on Raspberry Pi) or falls back to rules.
    """
    
    @staticmethod
    def _check_connection(model_name):
        """Verify Ollama is reachable and model exists. Never hangs."""
        if not OLLAMA_AVAILABLE:
            print(f"[LLM] _check_connection: ollama library not available")
            return False

        # 1. Quick TCP probe (instant fail if port closed)
        print(f"[LLM] _check_connection: TCP probe to 127.0.0.1:{OLLAMA_PORT}...")
        try:
            sock = socket.create_connection(("127.0.0.1", OLLAMA_PORT), timeout=CONNECTION_TIMEOUT)
            sock.close()
            print(f"[LLM] _check_connection: TCP probe OK")
        except (OSError, socket.timeout) as e:
            print(f"[LLM] _check_connection: TCP probe FAILED — {e}")
            return False

        # 2. List models (with hard timeout)
        print(f"[LLM] _check_connection: Listing models (timeout={CONNECTION_TIMEOUT}s)...")
        def _list_models():
            resp = ollama.list()
            # Handle both old dict format and new object format
            if hasattr(resp, 'models'):
                return resp.models  # New format: response object
            return resp.get('models', [])  # Old format: dict
        
        models = _run_with_timeout(_list_models, CONNECTION_TIMEOUT, default=None)
        if models is None:
            print(f"[LLM] _check_connection: ollama.list() failed or timed out")
            return False
        
        # Extract model names — handle both object (.model) and dict (['name']) formats
        model_names = []
        for m in models:
            if hasattr(m, 'model'):
                model_names.append(m.model)      # New: Model object
            elif isinstance(m, dict):
                model_names.append(m.get('name', m.get('model', '')))
            else:
                model_names.append(str(m))
        
        print(f"[LLM] _check_connection: Available models: {model_names}")
        
        found = any(name.startswith(model_name) for name in model_names)
        print(f"[LLM] _check_connection: Model '{model_name}' found = {found}")
        return found

    @staticmethod
    def generate_explanation_stream(blasts, model=DEFAULT_MODEL):
        """
        Generator that yields tokens for a real-time typing effect.
        Falls back to rule-based if Ollama is unavailable/slow.
        """
        print(f"\n[LLM] === generate_explanation_stream START ===")
        print(f"[LLM] Blast count: {len(blasts)}, Model: {model}")
        
        timestamp = datetime.now().strftime("%H:%M")
        yield f"[{timestamp} AI] "
        
        # 1. Quick Fallback for empty input
        if not blasts:
            print(f"[LLM] No blasts, returning normal message")
            yield "No blast cells detected. Peripheral blood smear appears normal."
            return

        # 2. Construct Prompt
        prompt = LLMGenerator._build_prompt(blasts)
        print(f"[LLM] Prompt constructed, checking connection...")

        # 3. Try Ollama (with aggressive fallback)
        used_rule_based = True
        t_start = time.time()

        if LLMGenerator._check_connection(model):
            print(f"[LLM] Connection OK ({time.time()-t_start:.1f}s). Starting chat stream...")
            try:
                t_chat = time.time()
                stream = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=True,
                    options={'num_predict': 100},  # Cap output for speed
                )
                print(f"[LLM] ollama.chat() returned stream object ({time.time()-t_chat:.1f}s)")

                token_count = 0
                for chunk in stream:
                    content = chunk['message']['content']
                    if content:
                        yield content
                        used_rule_based = False
                        token_count += 1
                
                print(f"[LLM] Stream complete: {token_count} tokens in {time.time()-t_chat:.1f}s")

            except Exception as e:
                print(f"[LLM] ERROR during streaming: {e}")
                traceback.print_exc()
        else:
            print(f"[LLM] Connection check FAILED ({time.time()-t_start:.1f}s) — using fallback")
        
        # 4. Fallback execution
        if used_rule_based:
            print(f"[LLM] Using RULE-BASED fallback")
            metrics = LLMGenerator._calculate_metrics(blasts)
            full_text = LLMGenerator._generate_rule_based(metrics, len(blasts))
            
            for word in full_text.split(' '):
                yield word + " "
                time.sleep(0.03)  # Fast typing
            
        print(f"[LLM] === generate_explanation_stream END ({time.time()-t_start:.1f}s total) ===\n")

    @staticmethod
    def generate_explanation(blasts, model=DEFAULT_MODEL):
        """Blocking version (legacy support)."""
        generator = LLMGenerator.generate_explanation_stream(blasts, model)
        return "".join(list(generator))

    @staticmethod
    def _calculate_metrics(blasts):
        if not blasts: return (0,0,0)
        n = len(blasts)
        return (
            sum(b['circularity'] for b in blasts) / n,
            sum(b['homogeneity'] for b in blasts) / n,
            sum(b['score'] for b in blasts) / n
        )

    @staticmethod
    def _build_prompt(blasts):
        """Build a data-grounded prompt with per-cell measurements."""
        count = len(blasts)
        
        # Per-cell table
        rows = []
        for b in blasts:
            rows.append(
                f"  Cell #{b['id']}: area={b.get('area','?')}px, "
                f"circularity={b['circularity']*100:.0f}%, "
                f"eccentricity={b.get('eccentricity',0):.2f}, "
                f"score={b['score']:.2f}"
            )
        cell_table = "\n".join(rows)
        
        # Averages
        avg_circ = sum(b['circularity'] for b in blasts) / count
        avg_score = sum(b['score'] for b in blasts) / count
        
        return f"""You are analyzing the output of an automated ALL (Acute Lymphoblastic Leukemia) detection system that processed a PERIPHERAL BLOOD SMEAR image.

The system detected {count} suspected blast cell(s). Here are the measured parameters for each:
{cell_table}

Averages: circularity={avg_circ*100:.0f}%, classifier_score={avg_score:.2f}

RULES:
- This is a PERIPHERAL BLOOD SMEAR, NOT a bone marrow biopsy.
- Only refer to the data above. Do NOT invent findings not listed.
- Write exactly 2 sentences: one describing what was found, one suggesting next steps.
- Use professional clinical tone. Be concise."""

    @staticmethod
    def _generate_rule_based(metrics, count):
        """Deterministic fallback logic."""
        avg_circ, avg_homo, avg_score = metrics
        
        parts = []
        if avg_circ > 0.75:
            parts.append(f"Cells display high circularity ({avg_circ*100:.0f}%), consistent with L1 lymphoblasts.")
        else:
            parts.append(f"Irregular nuclear contours observed (circularity {avg_circ*100:.0f}%).")
            
        if avg_score > 3.0:
            parts.append(f"High confidence detection (Score {avg_score:.2f}).")
            
        return " ".join(parts)

