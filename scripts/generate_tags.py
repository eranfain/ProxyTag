#!/usr/bin/env python3
"""
Generate tags for items using LLM prompts.

Supports multiple prompt strategies and datasets (Amazon Books, MovieLens).
Supports two providers:
  - Azure OpenAI (parallel real-time inference)
  - Groq (parallel real-time or batch API inference)

Includes cost estimation mode for Groq.
"""

import argparse
import os
import json
import math
import time
import pandas as pd
import tiktoken
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# Groq pricing per 1M tokens (as of 2025)
# Update these if pricing changes.
# ============================================================
GROQ_PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.2-3b-preview": {"input": 0.06, "output": 0.06},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.2-90b-vision-preview": {"input": 0.90, "output": 0.90},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "deepseek-r1-distill-llama-70b": {"input": 0.75, "output": 0.99},
    "qwen-qwq-32b": {"input": 0.29, "output": 0.39},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"input": 0.50, "output": 0.77},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}


def get_azure_client():
    """Initialize Azure OpenAI client from environment variables."""
    from openai import AzureOpenAI

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not api_key or not endpoint:
        raise ValueError(
            "Missing Azure OpenAI credentials. Set:\n"
            "  export AZURE_OPENAI_API_KEY=your_key\n"
            "  export AZURE_OPENAI_ENDPOINT=your_endpoint"
        )

    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )


def get_groq_client():
    """Initialize Groq client from environment variables."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Groq API key. Set:\n"
            "  export GROQ_API_KEY=your_key"
        )

    return Groq(api_key=api_key)


# ============================================================
# Prompt generation
# ============================================================

def generate_book_prompts():
    """
    Generate all 16 prompt configurations for book tagging.

    Each prompt differs along four axes:
      1. single vs multi-word tags
      2. use LLM knowledge base or not
      3. describe vs recommend perspective
      4. semantic focus (content vs mood/style)

    Returns:
        dict: Mapping from prompt name to prompt template
    """
    granularity = {
        "single": "Generate single-word tags only.",
        "multi": "Generate multi-word tags when needed to convey meaningful concepts."
    }

    knowledge = {
        "extract_only": (
            "Tags must be derived strictly from the provided book details. "
            "Do not use knowledge not grounded in the book details."
        ),
        "use_knowledge": (
            "You may use your external knowledge about the book "
            "to enrich the tags beyond what is explicitly stated in the provided details."
        )
    }

    intent = {
        "describe": (
            "Generate tags that best *describe* the book, its themes, concepts, and elements."
        ),
        "recommend": (
            "Generate tags that would help recommend this book to the right audience—"
            "focus on aspects that contribute to user preference and decision-making."
        )
    }

    focus = {
        "content": (
            "Prioritize genre, plot, world elements, setting, and concrete content-specific concepts. "
            "Mood/style tags are allowed but secondary."
        ),
        "mood_style": (
            "Prioritize emotional tone, atmosphere, pacing, and reading experience. "
            "Genre and plot tags are allowed but secondary."
        )
    }

    base = (
        "You are an intelligent tagging system for a book recommendation engine.\n"
        "Your task is to extract a list of meaningful and diverse tags from provided book details. "
        "These tags will be transformed into embeddings and fed into a model for recommendation.\n\n"
        "Instructions:\n"
        "- Return only a list of tags separated by '|'.\n"
        "- Output the tags only with no explanations, prefixes, or suffixes."
    )

    prompts = {}
    for g_key, g_text in granularity.items():
        for k_key, k_text in knowledge.items():
            for i_key, i_text in intent.items():
                for f_key, f_text in focus.items():
                    key = f"{g_key}_{k_key}_{i_key}_{f_key}"
                    prompt = (
                        f"{base}\n"
                        f"- {g_text}\n"
                        f"- {k_text}\n"
                        f"- {i_text}\n"
                        f"- {f_text}\n"
                        "\nProcess the following book:\n"
                        "Name: {title}\n"
                        "Genres: {genres}\n"
                        "Description: {text}\n"
                        "Tags:"
                    )
                    prompts[key] = prompt

    return prompts


def generate_movie_prompts():
    """
    Generate all 16 prompt configurations for movie tagging.
    Each prompt differs along four axes:
      1. single vs multi-word tags
      2. use LLM knowledge base or not
      3. describe vs recommend perspective
      4. semantic focus (content vs mood/style)
    """
    granularity = {
        "single": "Generate single-word tags only.",
        "multi": "Generate multi-word tags when needed to convey meaningful concepts."
    }

    knowledge = {
        "extract_only": (
            "Tags must be derived strictly from the provided movie description. "
            "Do not use knowledge not grounded in the text."
        ),
        "use_knowledge": (
            "You may use your external knowledge about the movie or its cultural context "
            "to enrich the tags beyond what is explicitly stated in the description."
        )
    }

    intent = {
        "describe": (
            "Generate tags that best *describe* the movie, its themes, concepts, and elements."
        ),
        "recommend": (
            "Generate tags that would help recommend this movie to the right audience—"
            "focus on aspects that contribute to user preference and decision-making."
        )
    }

    focus = {
        "content": (
            "Prioritize genre, plot, world elements, setting, and concrete content-specific concepts. "
            "Mood/style tags are allowed but secondary."
        ),
        "mood_style": (
            "Prioritize emotional tone, atmosphere, pacing, and viewing experience. "
            "Genre and plot tags are allowed but secondary."
        )
    }

    base = (
        "You are an intelligent tagging system for a movie recommendation engine.\n"
        "Your task is to extract a list of meaningful and diverse tags from a movie description. "
        "These tags will be transformed into embeddings and fed into a model for recommendation.\n\n"
        "Instructions:\n"
        "- Return only a list of tags separated by '|'.\n"
        "- Output the tags only with no explanations, prefixes, or suffixes."
    )

    prompts = {}
    for g_key, g_text in granularity.items():
        for k_key, k_text in knowledge.items():
            for i_key, i_text in intent.items():
                for f_key, f_text in focus.items():
                    key = f"{g_key}_{k_key}_{i_key}_{f_key}"
                    prompt = (
                        f"{base}\n"
                        f"- {g_text}\n"
                        f"- {k_text}\n"
                        f"- {i_text}\n"
                        f"- {f_text}\n"
                        "\nProcess the following movie:\n"
                        "Title: {title}\n"
                        "Genres: {genres}\n"
                        "Summary: {summary}\n"
                        "Tags:"
                    )
                    prompts[key] = prompt

    return prompts


# ============================================================
# Cost estimation
# ============================================================

def estimate_tokens(text, encoding_name="cl100k_base"):
    """Estimate token count for a string using tiktoken."""
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        return len(text.split()) * 1.3


def estimate_cost(
    df: pd.DataFrame,
    prompts_to_run: dict,
    model: str,
    dataset: str,
    title_col: str,
    description_col: str,
    genres_col: str = "genres",
    output_tokens_estimate: int = 100,
):
    """
    Estimate the cost of running all prompts over the dataset using Groq.

    Samples 200 items, formats each prompt template with actual data to
    measure input token counts, then extrapolates to the full dataset.

    Args:
        df: DataFrame with items
        prompts_to_run: Dict of prompt_name -> prompt_template
        model: Groq model name
        dataset: 'amazon-books' or 'amazon-movies'
        title_col: Column name for title
        description_col: Column name for description/summary
        genres_col: Column name for genres
        output_tokens_estimate: Estimated output tokens per request

    Returns:
        dict with cost breakdown
    """
    if model not in GROQ_PRICING:
        print(f"\nWarning: Model '{model}' not found in pricing table.")
        print(f"Available models with known pricing:")
        for m, p in sorted(GROQ_PRICING.items()):
            print(f"  {m}: ${p['input']}/M input, ${p['output']}/M output")
        print(f"\nUsing placeholder pricing of $0.50/M input, $0.70/M output")
        pricing = {"input": 0.50, "output": 0.70}
    else:
        pricing = GROQ_PRICING[model]

    n_items = len(df)
    n_prompts = len(prompts_to_run)

    # Sample items to estimate average input tokens
    sample_size = min(200, n_items)
    sample_df = df.sample(n=sample_size, random_state=42)

    total_input_tokens = 0
    for prompt_name, prompt_template in prompts_to_run.items():
        for _, row in sample_df.iterrows():
            title = str(row.get(title_col, "") or "")
            desc = str(row.get(description_col, "") or "")
            genres = str(row.get(genres_col, "") or "")

            if dataset == 'amazon-books':
                formatted = prompt_template.format(title=title, text=desc, genres=genres)
            else:
                formatted = prompt_template.format(title=title, summary=desc, genres=genres)

            total_input_tokens += estimate_tokens(formatted)

    avg_input_tokens_per_request = total_input_tokens / (sample_size * n_prompts)
    total_requests = n_items * n_prompts
    total_input_tokens_est = int(avg_input_tokens_per_request * total_requests)
    total_output_tokens_est = output_tokens_estimate * total_requests

    input_cost = (total_input_tokens_est / 1_000_000) * pricing["input"]
    output_cost = (total_output_tokens_est / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "pricing_input_per_M": pricing["input"],
        "pricing_output_per_M": pricing["output"],
        "n_items": n_items,
        "n_prompts": n_prompts,
        "total_requests": total_requests,
        "avg_input_tokens_per_request": int(avg_input_tokens_per_request),
        "output_tokens_per_request_estimate": output_tokens_estimate,
        "total_input_tokens": total_input_tokens_est,
        "total_output_tokens": total_output_tokens_est,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


# ============================================================
# Inference: Azure OpenAI (real-time parallel)
# ============================================================

def run_prompt_azure(
    df: pd.DataFrame,
    user_prompt: str,
    output_col_name: str,
    client,
    deployment: str,
    max_workers: int = 16,
    validate_json_output: bool = False
):
    """Run LLM prompt over DataFrame in parallel using Azure OpenAI."""
    def run_prompt(prompt_vars, verbose=False):
        try:
            _user_prompt = user_prompt.format(**prompt_vars)
            if verbose:
                print(_user_prompt)

            chat_prompt = [
                {"role": "user", "content": [{"type": "text", "text": _user_prompt}]}
            ]

            completion = client.chat.completions.create(
                model=deployment,
                messages=chat_prompt,
                max_completion_tokens=1024,
                temperature=0.7,
                stop=None,
                stream=False
            )

            return json.loads(completion.to_json())['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            return None

    def retry_if_issue(row):
        if row[output_col_name] is None:
            return run_prompt(dict(row))
        elif validate_json_output:
            try:
                json.loads(row[output_col_name])
                return row[output_col_name]
            except Exception:
                return run_prompt(dict(row))
        else:
            return row[output_col_name]

    df['dummy'] = None

    if output_col_name in df.columns:
        print(f"Column '{output_col_name}' exists, retrying None values only")
        tqdm.pandas()
        df[output_col_name] = df.progress_apply(retry_if_issue, axis=1)
    else:
        print(f"Running prompt '{output_col_name}' in parallel with {max_workers} workers")
        params = [dict(r) for _, r in df.iterrows()]
        out = [None] * len(params)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_prompt, p): i for i, p in enumerate(params)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating tags"):
                out[futures[fut]] = fut.result()

        df[output_col_name] = out

    df = df.drop(columns=["dummy"])
    return df


# ============================================================
# Inference: Groq (real-time parallel)
# ============================================================

def run_prompt_groq(
    df: pd.DataFrame,
    user_prompt: str,
    output_col_name: str,
    client,
    model: str,
    max_workers: int = 16,
    temperature: float = 0.7,
    validate_json_output: bool = False
):
    """Run LLM prompt over DataFrame in parallel using Groq."""
    def run_prompt(prompt_vars, verbose=False):
        try:
            _user_prompt = user_prompt.format(**prompt_vars)
            if verbose:
                print(_user_prompt)

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": _user_prompt}],
                model=model,
                temperature=temperature,
                max_completion_tokens=1024,
                reasoning_effort="low",
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
            return None

    def retry_if_issue(row):
        if row[output_col_name] is None:
            return run_prompt(dict(row))
        elif validate_json_output:
            try:
                json.loads(row[output_col_name])
                return row[output_col_name]
            except Exception:
                return run_prompt(dict(row))
        else:
            return row[output_col_name]

    df['dummy'] = None

    if output_col_name in df.columns:
        print(f"Column '{output_col_name}' exists, retrying None values only")
        tqdm.pandas()
        df[output_col_name] = df.progress_apply(retry_if_issue, axis=1)
    else:
        print(f"Running prompt '{output_col_name}' via Groq ({model}) with {max_workers} workers")
        params = [dict(r) for _, r in df.iterrows()]
        out = [None] * len(params)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_prompt, p): i for i, p in enumerate(params)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating tags"):
                out[futures[fut]] = fut.result()

        df[output_col_name] = out

    df = df.drop(columns=["dummy"])
    return df


# ============================================================
# Inference: Groq Batch API
# ============================================================

def submit_groq_batch(
    df: pd.DataFrame,
    user_prompt: str,
    output_col_name: str,
    client,
    model: str,
    output_dir: str,
    temperature: float = 0.7,
    max_rows_per_file: int = 45_000,
):
    """
    Submit a Groq batch job without waiting for completion.

    Returns dict with batch_ids, batch_dir, and n_items for later monitoring.
    """
    print(f"  Submitting batch for '{output_col_name}' via Groq Batch API ({model})")

    n_items = len(df)
    n_files = math.ceil(n_items / max_rows_per_file)
    batch_dir = os.path.join(output_dir, "batch_files")
    os.makedirs(batch_dir, exist_ok=True)

    file_paths = []
    for file_idx in range(n_files):
        start = file_idx * max_rows_per_file
        end = min(start + max_rows_per_file, n_items)
        output_path = os.path.join(batch_dir, f"{output_col_name}_{file_idx + 1}.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, (_, row) in enumerate(df.iloc[start:end].iterrows()):
                prompt_vars = dict(row)
                prompt_vars['dummy'] = None
                try:
                    formatted_prompt = user_prompt.format(**prompt_vars)
                except (KeyError, TypeError):
                    formatted_prompt = user_prompt

                record = {
                    "custom_id": f"request-{start + idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "temperature": temperature,
                        "max_completion_tokens": 1024,
                        "reasoning_effort": "low",
                        "messages": [
                            {"role": "user", "content": formatted_prompt}
                        ]
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        file_paths.append(output_path)
        print(f"    Wrote {end - start} rows to {output_path}")

    batch_ids = []
    log_file = os.path.join(output_dir, "batch_log.jsonl")
    for fp in file_paths:
        response = client.files.create(file=open(fp, "rb"), purpose="batch")
        print(f"    Uploaded: {response.id} ({response.bytes} bytes)")

        batch_response = client.batches.create(
            completion_window="72h",
            endpoint="/v1/chat/completions",
            input_file_id=response.id,
        )
        batch_ids.append(batch_response.id)
        print(f"    Batch created: {batch_response.id}")

        with open(log_file, "a") as lf:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model,
                "output_col_name": output_col_name,
                "input_file_id": response.id,
                "batch_id": batch_response.id,
                "local_jsonl": fp,
                "n_items": n_items,
            }
            lf.write(json.dumps(log_entry) + "\n")

    return {
        "batch_ids": batch_ids,
        "batch_dir": batch_dir,
        "n_items": n_items,
    }


def download_batch_results(client, batch_ids, completed_statuses, batch_dir, output_col_name, n_items):
    """Download and parse results from completed Groq batches."""
    results = {}
    for bid in batch_ids:
        status = completed_statuses[bid]
        if status.status != "completed" or not status.output_file_id:
            continue

        content = client.files.content(status.output_file_id)
        output_path = os.path.join(batch_dir, f"{output_col_name}_{bid}_output.jsonl")
        content.write_to_file(output_path)

        with open(output_path, "r") as f:
            for line in f:
                record = json.loads(line)
                idx = int(record["custom_id"].replace("request-", ""))
                try:
                    text = record["response"]["body"]["choices"][0]["message"]["content"]
                    results[idx] = text
                except (KeyError, IndexError, TypeError):
                    results[idx] = None

    out = [results.get(i, None) for i in range(n_items)]
    n_success = sum(1 for v in out if v is not None)
    print(f"  Results for '{output_col_name}': {n_success}/{n_items} successful")
    return out


def monitor_and_save_batches(client, jobs, df, item_id_col):
    """
    Monitor all submitted batch jobs, downloading and saving results as they complete.

    Args:
        client: Groq client
        jobs: list of dicts with keys: prompt_name, batch_info (from submit_groq_batch),
              output_file, existing_tags, df_to_generate
        df: original full DataFrame (for merging results)
        item_id_col: item ID column name
    """
    all_batch_ids = {}
    for job in jobs:
        for bid in job["batch_info"]["batch_ids"]:
            all_batch_ids[bid] = job["prompt_name"]

    completed_batches = {}
    saved_prompts = set()

    print(f"\nMonitoring {len(all_batch_ids)} batch(es) across {len(jobs)} prompt(s)...")
    print()

    latest_status = {}

    while len(completed_batches) < len(all_batch_ids):
        for bid in all_batch_ids:
            if bid in completed_batches:
                continue
            try:
                status = client.batches.retrieve(bid)
                latest_status[bid] = status
            except Exception as e:
                print(f"  Poll error for {bid} (will retry): {e}")
                continue
            if status.status in ("completed", "failed", "cancelled", "expired"):
                completed_batches[bid] = status

        print(f"[{time.strftime('%H:%M:%S')}] Batch status:")
        for job in jobs:
            prompt_name = job["prompt_name"]
            batch_ids = job["batch_info"]["batch_ids"]
            if prompt_name in saved_prompts:
                print(f"  {prompt_name}: SAVED")
                continue

            statuses = []
            for bid in batch_ids:
                if bid in completed_batches:
                    statuses.append(completed_batches[bid].status)
                elif bid in latest_status:
                    s = latest_status[bid]
                    counts = s.request_counts
                    statuses.append(f"{s.status} ({counts.completed}/{counts.total})")
                else:
                    statuses.append("pending")
            print(f"  {prompt_name}: {', '.join(statuses)}")

        # Check if any prompt has all its batches done and hasn't been saved yet
        for job in jobs:
            prompt_name = job["prompt_name"]
            if prompt_name in saved_prompts:
                continue

            batch_ids = job["batch_info"]["batch_ids"]
            all_done = all(bid in completed_batches for bid in batch_ids)
            if not all_done:
                continue

            batch_statuses = {bid: completed_batches[bid] for bid in batch_ids}
            out = download_batch_results(
                client, batch_ids, batch_statuses,
                job["batch_info"]["batch_dir"],
                prompt_name,
                job["batch_info"]["n_items"],
            )

            # Merge new results with existing tags and save
            new_tags = pd.Series(out, index=job["df_to_generate"][item_id_col].values)
            existing_tags = job["existing_tags"]
            if existing_tags is not None:
                combined_tags = existing_tags.copy()
                combined_tags.update(new_tags)
                output_df = df[[item_id_col]].copy()
                output_df['tags'] = output_df[item_id_col].map(combined_tags)
            else:
                output_df = df[[item_id_col]].copy()
                output_df['tags'] = output_df[item_id_col].map(new_tags)

            output_df.to_parquet(job["output_file"], index=False)
            saved_prompts.add(prompt_name)

            n_with_tags = output_df['tags'].notna().sum()
            print(f"  Saved {job['output_file']}: {n_with_tags:,}/{len(output_df):,} ({n_with_tags/len(output_df):.1%})")

        print()

        if len(completed_batches) < len(all_batch_ids):
            time.sleep(30)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate tags using LLM prompts")

    # Input/output
    parser.add_argument('--input_path', type=str, required=True,
                        help='Input parquet file with item data')
    parser.add_argument('--output_dir', type=str, default='data/tags',
                        help='Base output directory (default: data/tags). '
                             'Files saved as {output_dir}/{dataset}/{split}/{variant}_{model}.parquet')
    parser.add_argument('--split', type=str, required=True,
                        choices=['full', 'panel'],
                        help='Whether generating for the full dataset or a panel subset')

    # Dataset type
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['amazon-books', 'amazon-movies'],
                        help='Dataset type (determines column names and prompts)')

    # Column mapping
    parser.add_argument('--item_id_col', type=str, required=True,
                        help='Item ID column name (e.g., asin, movieId)')
    parser.add_argument('--title_col', type=str, default='title',
                        help='Title column name')
    parser.add_argument('--description_col', type=str, default=None,
                        help='Description column name (required for books, optional for movies)')
    parser.add_argument('--genres_col', type=str, default='genres',
                        help='Genres column name (for movies)')

    # Prompts
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                        help='Specific prompts to run (default: all for dataset)')
    parser.add_argument('--list_prompts', action='store_true',
                        help='List available prompts and exit')

    # Provider
    parser.add_argument('--provider', type=str, default='groq',
                        choices=['azure', 'groq'],
                        help='LLM provider (default: groq)')

    # Azure-specific
    parser.add_argument('--deployment', type=str, default='gpt-5-mini',
                        help='Azure OpenAI deployment name')

    # Groq-specific
    parser.add_argument('--model', type=str, default='llama-3.1-8b-instant',
                        help='Groq model name (default: llama-3.1-8b-instant)')
    parser.add_argument('--groq_mode', type=str, default='batch',
                        choices=['realtime', 'batch'],
                        help='Groq inference mode: realtime (parallel) or batch API (default: batch)')

    # Cost estimation
    parser.add_argument('--estimate_cost', action='store_true',
                        help='Estimate cost without running inference (Groq only)')
    parser.add_argument('--output_tokens_estimate', type=int, default=100,
                        help='Estimated output tokens per request for cost estimation')

    # Execution
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Number of parallel workers (realtime mode)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Generation temperature')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of items (for testing)')

    args = parser.parse_args()

    # Load prompts
    if args.dataset == 'amazon-books':
        all_prompts = generate_book_prompts()
        required_cols = [args.title_col]
        if args.description_col:
            required_cols.append(args.description_col)
        else:
            args.description_col = 'description'
            required_cols.append('description')
    else:  # amazon-movies or any movie dataset
        all_prompts = generate_movie_prompts()
        required_cols = [args.title_col]
        if not args.description_col:
            args.description_col = 'overview'

    # List prompts if requested
    if args.list_prompts:
        print(f"\nAvailable prompts for {args.dataset}:")
        for i, name in enumerate(sorted(all_prompts.keys()), 1):
            print(f"  {i}. {name}")
        print(f"\nTotal: {len(all_prompts)} prompts")
        return

    # Select prompts to run
    if args.prompts:
        prompts_to_run = {k: v for k, v in all_prompts.items() if k in args.prompts}
        if not prompts_to_run:
            print(f"Error: None of the specified prompts found. Use --list_prompts to see available prompts.")
            return
    else:
        prompts_to_run = all_prompts

    # Load data
    print(f"Loading items from: {args.input_path}")
    df = pd.read_parquet(args.input_path)

    # Validate columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Limit if requested
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} items for testing")

    print(f"Items: {len(df):,}")
    print(f"\nFirst rows preview:")
    print(df.head(2).to_string())
    print()

    # ========================================
    # Cost estimation mode
    # ========================================
    if args.estimate_cost:
        print("\n" + "=" * 80)
        print("COST ESTIMATION (Groq)")
        print("=" * 80)

        models_to_estimate = [args.model]
        # If the user just wants a quick comparison, also show a second model
        # Automatically pair small with a big model for comparison
        small_models = ["llama-3.1-8b-instant", "llama-3.2-3b-preview", "gemma2-9b-it"]
        big_models = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "deepseek-r1-distill-llama-70b"]

        if args.model in small_models:
            # Also estimate with first available big model
            for bm in big_models:
                if bm != args.model:
                    models_to_estimate.append(bm)
                    break
        elif args.model in big_models:
            # Also estimate with first available small model
            for sm in small_models:
                if sm != args.model:
                    models_to_estimate.append(sm)
                    break

        for model_name in models_to_estimate:
            est = estimate_cost(
                df=df,
                prompts_to_run=prompts_to_run,
                model=model_name,
                dataset=args.dataset,
                title_col=args.title_col,
                description_col=args.description_col,
                genres_col=args.genres_col,
                output_tokens_estimate=args.output_tokens_estimate,
            )

            print(f"\n--- Model: {est['model']} ---")
            print(f"  Pricing: ${est['pricing_input_per_M']}/M input, ${est['pricing_output_per_M']}/M output")
            print(f"  Items: {est['n_items']:,}")
            print(f"  Prompts: {est['n_prompts']}")
            print(f"  Total requests: {est['total_requests']:,}")
            print(f"  Avg input tokens/request: {est['avg_input_tokens_per_request']}")
            print(f"  Est output tokens/request: {est['output_tokens_per_request_estimate']}")
            print(f"  Total input tokens: {est['total_input_tokens']:,}")
            print(f"  Total output tokens: {est['total_output_tokens']:,}")
            print(f"  ---")
            print(f"  Input cost:  ${est['input_cost_usd']:.4f}")
            print(f"  Output cost: ${est['output_cost_usd']:.4f}")
            print(f"  TOTAL COST:  ${est['total_cost_usd']:.4f}")

        print("\n" + "=" * 80)
        print("Note: Output tokens are estimated. Actual cost may vary.")
        print("Batch API pricing may differ (often 50% discount).")
        print("=" * 80)
        return

    # ========================================
    # Generation mode
    # ========================================
    model_name = args.model if args.provider == 'groq' else args.deployment
    safe_model_name = model_name.replace("/", "-")
    dataset_output_dir = os.path.join(args.output_dir, args.dataset, args.split)
    os.makedirs(dataset_output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("TAG GENERATION")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Provider: {args.provider}")
    if args.provider == 'groq':
        print(f"Model: {args.model}")
        print(f"Mode: {args.groq_mode}")
    else:
        print(f"Deployment: {args.deployment}")
    print(f"Input: {args.input_path}")
    print(f"Output dir: {dataset_output_dir}")
    print(f"Prompts to run: {len(prompts_to_run)}")
    print()

    # Initialize client
    if args.provider == 'azure':
        try:
            client = get_azure_client()
            print("Azure OpenAI client initialized")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        try:
            client = get_groq_client()
            print("Groq client initialized")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # ========================================
    # Phase 1: Identify missing tags
    # ========================================
    print("\n" + "=" * 80)
    print("PHASE 1: Checking existing tags")
    print("=" * 80)

    pending_prompts = []
    n_skipped = 0

    for prompt_name, prompt_template in prompts_to_run.items():
        output_file = os.path.join(dataset_output_dir, f'{prompt_name}_{safe_model_name}.parquet')

        existing_tags = None
        if os.path.exists(output_file):
            existing_df = pd.read_parquet(output_file)
            existing_tags = existing_df.set_index(args.item_id_col)['tags']
            n_existing = existing_tags.notna().sum()
            n_total = len(df)
            all_ids = set(df[args.item_id_col])
            covered_ids = set(existing_tags[existing_tags.notna()].index)
            missing_ids = all_ids - covered_ids
            if not missing_ids:
                print(f"  {prompt_name}: complete ({n_total} items)")
                n_skipped += 1
                continue
            print(f"  {prompt_name}: {n_existing}/{n_total} exist, {len(missing_ids)} missing")
            df_to_generate = df[df[args.item_id_col].isin(missing_ids)]
        else:
            print(f"  {prompt_name}: no existing file")
            df_to_generate = df

        # Prepare DataFrame with correct column names for prompt
        df_prompt = df_to_generate.copy()
        if args.dataset == 'amazon-books':
            df_prompt['title'] = df_prompt[args.title_col]
            df_prompt['text'] = df_prompt[args.description_col]
            if args.genres_col in df_prompt.columns:
                df_prompt['genres'] = df_prompt[args.genres_col].fillna("")
            else:
                df_prompt['genres'] = ""
        else:
            df_prompt['title'] = df_prompt[args.title_col]
            if args.description_col in df_prompt.columns:
                df_prompt['summary'] = df_prompt[args.description_col].fillna("")
            else:
                df_prompt['summary'] = ""
            if args.genres_col in df_prompt.columns:
                df_prompt['genres'] = df_prompt[args.genres_col].fillna("")
            else:
                df_prompt['genres'] = ""

        pending_prompts.append({
            "prompt_name": prompt_name,
            "prompt_template": prompt_template,
            "df_prompt": df_prompt,
            "df_to_generate": df_to_generate,
            "existing_tags": existing_tags,
            "output_file": output_file,
        })

    print(f"\nSummary: {n_skipped} complete, {len(pending_prompts)} need generation")

    if not pending_prompts:
        print("\nAll prompts already complete!")
        return

    # ========================================
    # Phase 2 & 3: Submit and monitor
    # ========================================
    use_batch = args.provider == 'groq' and args.groq_mode == 'batch'

    if use_batch:
        # Batch mode: submit all jobs, then monitor concurrently
        print("\n" + "=" * 80)
        print(f"PHASE 2: Submitting {len(pending_prompts)} batch job(s)")
        print("=" * 80)

        jobs = []
        for item in pending_prompts:
            batch_output_dir = os.path.join(
                dataset_output_dir, 'batch_files',
                f'{item["prompt_name"]}_{safe_model_name}'
            )
            os.makedirs(batch_output_dir, exist_ok=True)

            batch_info = submit_groq_batch(
                item["df_prompt"], item["prompt_template"], 'tags',
                client, args.model, batch_output_dir, args.temperature
            )
            jobs.append({
                "prompt_name": item["prompt_name"],
                "batch_info": batch_info,
                "output_file": item["output_file"],
                "existing_tags": item["existing_tags"],
                "df_to_generate": item["df_to_generate"],
            })

        total_batches = sum(len(j["batch_info"]["batch_ids"]) for j in jobs)
        print(f"\nSubmitted {total_batches} batch(es) across {len(jobs)} prompt(s)")

        print("\n" + "=" * 80)
        print("PHASE 3: Monitoring batches")
        print("=" * 80)

        monitor_and_save_batches(client, jobs, df, args.item_id_col)

    else:
        # Real-time mode: run each prompt sequentially (already parallelized at request level)
        for i, item in enumerate(pending_prompts, 1):
            print(f"\n{'=' * 80}")
            print(f"Prompt {i}/{len(pending_prompts)}: {item['prompt_name']}")
            print(f"{'=' * 80}")

            df_prompt = item["df_prompt"]
            if args.provider == 'azure':
                df_prompt = run_prompt_azure(
                    df_prompt, item["prompt_template"], 'tags',
                    client, args.deployment, args.max_workers
                )
            else:
                df_prompt = run_prompt_groq(
                    df_prompt, item["prompt_template"], 'tags',
                    client, args.model, args.max_workers, args.temperature
                )

            new_tags = df_prompt.set_index(args.item_id_col)['tags']
            existing_tags = item["existing_tags"]
            if existing_tags is not None:
                combined_tags = existing_tags.copy()
                combined_tags.update(new_tags)
                output_df = df[[args.item_id_col]].copy()
                output_df['tags'] = output_df[args.item_id_col].map(combined_tags)
            else:
                output_df = df[[args.item_id_col]].copy()
                output_df['tags'] = output_df[args.item_id_col].map(new_tags)

            output_df.to_parquet(item["output_file"], index=False)

            n_with_tags = output_df['tags'].notna().sum()
            print(f"\nSaved: {item['output_file']}")
            print(f"  Items with tags: {n_with_tags:,} / {len(output_df):,} ({n_with_tags / len(output_df):.1%})")

    print("\n" + "=" * 80)
    print("TAG GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated tag files in: {dataset_output_dir}")
    print("\nNext steps:")
    print(f"  1. Create panel: python -m proxytag.analysis.create_panel_files --dataset_dir data/{args.dataset}")
    print(f"  2. Evaluate: python -m proxytag.analysis.proxy_metric --train_path data/{args.dataset}/interactions/train.parquet --items_data_path <panel_tags.parquet> --use_interaction_cf")


if __name__ == "__main__":
    main()
