#!/usr/bin/env python3
"""
Generate tags for items using LLM prompts.

Supports multiple prompt strategies and datasets (Amazon Books, MovieLens).
Uses Azure OpenAI for parallel tag generation.
"""

import argparse
import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_azure_client():
    """Initialize Azure OpenAI client from environment variables."""
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
        "- Include any of the following tag aspects: Genre and format tags; Content-specific nouns; "
        "Emotional or mood tags; Thematic or psychological concepts; Cultural or temporal tags.\n"
        "- Avoid generic terms.\n"
        "- Do not include character or brand names unless iconic or crucial to the item."
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

    # Axis options
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
        "- Include any of the following tag aspects: Genre and format tags; Content-specific nouns; "
        "Emotional or mood tags; Thematic or psychological concepts; Cultural or temporal tags.\n"
        "- Avoid generic terms.\n"
        "- Do not include character or brand names unless iconic or crucial to the item.\n"
        "- Tags should reflect the semantic meaning of the movie."
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
                        "Summary: {summary}\n"
                        "Tags:"
                    )
                    prompts[key] = prompt

    return prompts


def run_prompt_over_df(
    df: pd.DataFrame,
    user_prompt: str,
    output_col_name: str,
    client: AzureOpenAI,
    deployment: str,
    max_workers: int = 16,
    validate_json_output: bool = False
):
    """
    Run LLM prompt over DataFrame in parallel.

    Args:
        df: DataFrame with item data
        user_prompt: Prompt template with {column_name} placeholders
        output_col_name: Name for output column
        client: Azure OpenAI client
        deployment: Azure deployment name
        max_workers: Number of parallel workers
        validate_json_output: Retry if output is invalid JSON

    Returns:
        DataFrame with new output column
    """
    def run_prompt(prompt_vars, verbose=False):
        try:
            # Format prompt with variables
            _user_prompt = user_prompt.format(**prompt_vars)

            if verbose:
                print(_user_prompt)

            # Build chat messages
            chat_prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _user_prompt
                        }
                    ]
                }
            ]

            # Generate completion
            completion = client.chat.completions.create(
                model=deployment,
                messages=chat_prompt,
                max_completion_tokens=1024,
                stop=None,
                stream=False
            )

            return json.loads(completion.to_json())['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            return None

    def retry_if_issue(row):
        """Retry failed or invalid outputs."""
        if row[output_col_name] is None:
            return run_prompt(dict(row))
        elif validate_json_output:
            try:
                json.loads(row[output_col_name])
                return row[output_col_name]
            except:
                return run_prompt(dict(row))
        else:
            return row[output_col_name]

    df['dummy'] = None

    if output_col_name in df.columns:
        # Retry mode - only process None values
        print(f"Column '{output_col_name}' exists, retrying None values only")
        tqdm.pandas()
        df[output_col_name] = df.progress_apply(retry_if_issue, axis=1)
    else:
        # Parallel execution mode
        print(f"Running prompt '{output_col_name}' in parallel with {max_workers} workers")
        params = [dict(r) for i, r in df.iterrows()]
        out = [None] * len(params)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_prompt, p): i for i, p in enumerate(params)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating tags"):
                out[futures[fut]] = fut.result()

        df[output_col_name] = out

    df = df.drop(columns=["dummy"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate tags using LLM prompts")

    # Input/output
    parser.add_argument('--input_path', type=str, required=True,
                        help='Input parquet file with item data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for generated tag files')

    # Dataset type
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['amazon-books', 'movielens'],
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

    # Azure OpenAI
    parser.add_argument('--deployment', type=str, default='gpt-5-mini',
                        help='Azure OpenAI deployment name')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Number of parallel workers')

    # Other
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
            print("Warning: --description_col not specified, using 'description'")
            args.description_col = 'description'
            required_cols.append('description')
    else:  # movielens
        all_prompts = generate_movie_prompts()
        required_cols = [args.title_col]
        if not args.description_col:
            args.description_col = 'description'  # May not exist, that's ok

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

    print("="*80)
    print("TAG GENERATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_dir}")
    print(f"Prompts to run: {len(prompts_to_run)}")
    print()

    # Load data
    print(f"Loading items from: {args.input_path}")
    df = pd.read_parquet(args.input_path)

    # Validate columns
    missing_cols = [c for c in required_cols if c not in df.columns and c != args.description_col]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Limit if requested
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} items for testing")

    print(f"Items: {len(df):,}")
    print()

    # Initialize Azure client
    try:
        client = get_azure_client()
        print("✓ Azure OpenAI client initialized")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run each prompt
    for i, (prompt_name, prompt_template) in enumerate(prompts_to_run.items(), 1):
        print(f"\n{'='*80}")
        print(f"Prompt {i}/{len(prompts_to_run)}: {prompt_name}")
        print(f"{'='*80}")

        # Prepare DataFrame with correct column names for prompt
        if args.dataset == 'amazon-books':
            df_prompt = df.copy()
            df_prompt['title'] = df_prompt[args.title_col]
            df_prompt['text'] = df_prompt[args.description_col]
            df_prompt['genres'] = df_prompt[args.genres_col]
        else:  # movielens
            df_prompt = df.copy()
            df_prompt['title'] = df_prompt[args.title_col]
            if args.description_col in df_prompt.columns:
                df_prompt['summary'] = df_prompt[args.description_col]
            else:
                df_prompt['summary'] = ""  # Fallback

        # Run prompt
        df_prompt = run_prompt_over_df(
            df_prompt,
            prompt_template,
            'tags',
            client,
            args.deployment,
            args.max_workers
        )

        # Save output
        output_file = os.path.join(args.output_dir, f'tags_{prompt_name}.parquet')
        output_df = df_prompt[[args.item_id_col, 'tags']]
        output_df.to_parquet(output_file, index=False)

        # Statistics
        n_with_tags = output_df['tags'].notna().sum()
        print(f"\n✓ Saved: {output_file}")
        print(f"  Items with tags: {n_with_tags:,} / {len(output_df):,} ({n_with_tags/len(output_df):.1%})")

        if n_with_tags > 0:
            sample_tags = output_df[output_df['tags'].notna()]['tags'].iloc[0]
            n_tags = len([t for t in sample_tags.split('|') if t.strip()])
            print(f"  Sample: {n_tags} tags - {sample_tags[:100]}...")

    print("\n" + "="*80)
    print("✓ TAG GENERATION COMPLETE")
    print("="*80)
    print(f"\nGenerated {len(prompts_to_run)} tag files in: {args.output_dir}")
    print("\nNext steps:")
    print(f"  1. Place files in: data/{args.dataset}/tags/full/")
    print(f"  2. Create panel: python create_panel_files.py --dataset_dir data/{args.dataset}")
    print(f"  3. Evaluate: python compare_tag_methods.py ...")


if __name__ == "__main__":
    main()
