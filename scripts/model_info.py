"""Debug: check that HF repos exist and summarize architecture / chat-template features (needs --net).

  ./run.sh --net python scripts/model_info.py Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3.5-4B
"""
import json
import sys

from huggingface_hub import HfApi, hf_hub_download

for repo in sys.argv[1:]:
    try:
        HfApi().model_info(repo)
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        tc = cfg.get("text_config", cfg)
        print(f"== {repo}\n   archs={cfg.get('architectures')} model_type={cfg.get('model_type')} "
              f"layers={tc.get('num_hidden_layers')} hidden={tc.get('hidden_size')} "
              f"layer_types={sorted(set(tc.get('layer_types') or []))} transformers={cfg.get('transformers_version')}")
        tpl = json.load(open(hf_hub_download(repo, "tokenizer_config.json"))).get("chat_template") or ""
        if not tpl:
            try:
                tpl = open(hf_hub_download(repo, "chat_template.jinja")).read()
            except Exception:  # noqa: BLE001
                pass
        print(f"   template: tools={'tools' in tpl} enable_thinking={'enable_thinking' in tpl} think={'<think>' in tpl}")
    except Exception as e:  # noqa: BLE001
        print(f"== {repo}: {type(e).__name__}: {str(e)[:300]}")
