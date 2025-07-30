# Inference-Pipeline-with-Custom-Transformer

# Introduction

This repository provides an end-to-end inference pipeline for processing free-form medical questions. It uses a custom PubMedBERT-based classifier to identify the topic of a query, fetches relevant context from Wikipedia, and generates concise, empathetic responses via a fine-tuned physician-style transformer.

# Features

Topic Classification: PubMedBERT with frozen label embeddings maps raw medical queries to topic labels.

Context Retrieval: Fetches introduction and key sections (Signs and symptoms, Causes, Diagnosis, Prevention, Treatment) from Wikipedia using the MediaWiki API.

Clean & Concatenate: Cleans HTML and wiki markup, then constructs a structured prompt combining context and user question.

Physician-like Generation: Uses a fine-tuned causal LM (FP16-enabled) to generate clinically accurate, empathetic answers.

Interactive CLI: REPL interface for continuous querying, with graceful exit commands (exit, quit, q).

GPU/CPU Agnostic: Supports mixed-precision inference on CUDA-enabled GPUs or falls back to CPU.

# Prerequisites

Python 3.8 or higher

pip package manager

(Optional) CUDA-enabled GPU for faster inference

# Configuration

ARTIFACT_DIR: Path to the directory containing label_embs.pt, id2label.pkl, and classifier.pt for the PubMedBERT classifier.

PHYS_MODEL_DIR: Directory of the fine-tuned physician-style transformer model.

WIKI_API: URL of the MediaWiki API endpoint (defaults to https://en.wikipedia.org/w/api.php).

Adjust these variables at the top of inference_pipeline_with_custom_transformer.py as needed.
