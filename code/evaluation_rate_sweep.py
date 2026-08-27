"""Evaluation-only greedy/Top-K rate sweep for existing MINERVA checkpoints."""

from __future__ import absolute_import, division

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import tensorflow as tf

from minerva.code.data.embedding_server import EmbeddingServer
from minerva.code.data.setup import set_seeds
from minerva.code.model.trainer import TrainerNLQ
from minerva.code.options import read_options

from policy_entropy.rate_eval import (
    build_graph_structural_relation_prior,
    evaluate_rate_mode,
    recover_raw_action_counts,
)
from policy_entropy.rate_plotting import (
    generate_rate_sweep_plots,
    save_rate_sweep_outputs,
)


logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"yes", "true", "t", "y", "1"}:
        return True
    if normalized in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def preparse_rate_arguments(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    """Consume rate-only options and return untouched arguments for MINERVA."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--rate_top_k", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--rate_include_unrestricted", type=_str2bool, default=True)
    parser.add_argument("--rate_seed", type=int, default=42)
    parser.add_argument("--rate_compute_task_agnostic", type=_str2bool, default=True)
    parser.add_argument("--rate_compute_truncation_diagnostics", type=_str2bool, default=True)
    parser.add_argument("--rate_prior_alpha", type=float, default=1.0)
    parser.add_argument("--rate_max_batches", type=int, default=None)
    rate_args, minerva_args = parser.parse_known_args(argv)

    if not rate_args.rate_top_k:
        parser.error("--rate_top_k must contain at least one support size.")
    invalid = [
        top_k
        for top_k in rate_args.rate_top_k
        if top_k < 1 or (top_k & (top_k - 1)) != 0
    ]
    if invalid:
        parser.error(f"Top-K support sizes must be powers of two; received {invalid}.")
    if rate_args.rate_prior_alpha <= 0.0:
        parser.error("--rate_prior_alpha must be positive.")
    if rate_args.rate_max_batches is not None and rate_args.rate_max_batches < 1:
        parser.error("--rate_max_batches must be positive when provided.")
    return rate_args, minerva_args


def _dataset_name(options: Dict[str, Any]) -> str:
    qa_stem = os.path.splitext(os.path.basename(options.get("raw_QAData_path", "")))[0]
    known = {
        "mquake_sa_qa_nhop": "mquake_st_single",
        "mquake_ma_qa_nhop": "mquake_st_multi",
        "metaqa_qa_nhop": "metaqa",
        "kinship_qa_nhop": "kinshiphinton",
    }
    if qa_stem in known:
        return known[qa_stem]
    return os.path.basename(os.path.normpath(options.get("data_input_dir", "test")))


def _git_revision(path: str) -> Any:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _checkpoint_identity(checkpoint_prefix: str) -> Dict[str, Any]:
    files = []
    for path in sorted(glob.glob(checkpoint_prefix + ".*")):
        stat = os.stat(path)
        files.append(
            {
                "path": path,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return {"prefix": checkpoint_prefix, "files": files}


def _build_trainer(options: Dict[str, Any], entity_vocab, relation_vocab, embedding_server):
    return TrainerNLQ(
        batch_size=options["batch_size"],
        test_batch_size=options["test_batch_size"],
        num_rollouts=options["num_rollouts"],
        test_rollouts=options["test_rollouts"],
        positive_reward=options["positive_reward"],
        negative_reward=options["negative_reward"],
        path_length=options["path_length"],
        data_input_dir=options["data_input_dir"],
        question_tokenizer_name=options["question_tokenizer_name"],
        question_format=options["question_format"],
        cached_QAMetaData_path=options["cached_QAMetaData_path"],
        raw_QAData_path=options["raw_QAData_path"],
        force_data_prepro=False,
        evaluate_paraphrases=options["evaluate_paraphrases"],
        max_num_actions=options["max_num_actions"],
        embedding_size=options["embedding_size"],
        hidden_size=options["hidden_size"],
        use_entity_embeddings=options["use_entity_embeddings"],
        train_entity_embeddings=options["train_entity_embeddings"],
        train_relation_embeddings=options["train_relation_embeddings"],
        LSTM_layers=options["LSTM_layers"],
        projection_adapter=options["projection_adapter"],
        projection_layers=options["projection_layers"],
        projection_hidden=options["projection_hidden"],
        learning_rate=options["learning_rate"],
        grad_clip_norm=options["grad_clip_norm"],
        gamma=options["gamma"],
        Lambda=options["Lambda"],
        beta=options["beta"],
        total_iterations=options["total_iterations"],
        eval_every=options["eval_every"],
        output_dir=options["output_dir"],
        model_dir=options["model_dir"],
        path_logger_file=options["path_logger_file"],
        pool=options["pool"],
        use_beam=options["use_beam"],
        seed=options["seed"],
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        use_weighted_hop_sampling=options["use_weighted_hop_sampling"],
        use_full_graph=options["use_full_graph"],
        use_directed_graph=options["use_directed_graph"],
        use_stop_signal=options["use_stop_signal"],
        use_restart_signal=options["use_restart_signal"],
        stop_signal_reward=options["stop_signal_reward"],
        stop_signal_penalty=options["stop_signal_penalty"],
        length_penalty=options["length_penalty"],
        path_segment_policy=options["path_segment_policy"],
        embedding_server=embedding_server,
        use_wandb=False,
    )


def main() -> None:
    rate_args, minerva_args = preparse_rate_arguments(sys.argv[1:])
    sys.argv = [sys.argv[0]] + minerva_args
    options = read_options()
    if options["use_beam"]:
        raise ValueError("Rate-sweep experiments require use_beam=False for comparability.")
    if options["pool"] != "max":
        raise ValueError("The P0 rate evaluator reproduces MINERVA pool='max'; current pool differs.")

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: [ %(message)s ]", "%Y/%m/%d %I:%M:%S %p")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    logfile = logging.FileHandler(options["log_file_name"], "w")
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    with open(os.path.join(options["vocab_dir"], "relation_vocab.json"), encoding="utf-8") as f:
        relation_vocab = json.load(f)
    with open(os.path.join(options["vocab_dir"], "entity_vocab.json"), encoding="utf-8") as f:
        entity_vocab = json.load(f)

    set_seeds(options["seed"])
    embedding_server = EmbeddingServer(options["question_tokenizer_name"])
    trainer = None
    try:
        trainer = _build_trainer(options, entity_vocab, relation_vocab, embedding_server)
        relation_prior = None
        prior_metadata = {"enabled": False}
        if rate_args.rate_compute_task_agnostic:
            relation_prior, prior_metadata = build_graph_structural_relation_prior(
                trainer, alpha=rate_args.rate_prior_alpha
            )
            prior_metadata["enabled"] = True
            prior_metadata["uses_full_graph"] = bool(options["use_full_graph"])

        raw_action_counts = None
        raw_action_metadata = {"available": False, "limitation": "Not requested."}
        if rate_args.rate_compute_truncation_diagnostics:
            raw_action_counts, raw_action_metadata = recover_raw_action_counts(
                trainer.environment.grapher
            )

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = False
        config.log_device_placement = False
        config.allow_soft_placement = True

        summaries = []
        with tf.compat.v1.Session(config=config) as sess:
            set_seeds(options["seed"])
            trainer.initialize(restore=options["model_load_dir"], sess=sess)

            requested_modes = [("greedy", None)]
            requested_modes.extend(("topk", top_k) for top_k in rate_args.rate_top_k)
            if rate_args.rate_include_unrestricted:
                # Run this last: NumPy override modes do not execute TF's categorical sampler.
                requested_modes.append(("tf_policy", None))

            for action_mode, top_k in requested_modes:
                summary = evaluate_rate_mode(
                    trainer=trainer,
                    sess=sess,
                    action_mode=action_mode,
                    top_k=top_k,
                    seed=rate_args.rate_seed,
                    relation_prior=relation_prior,
                    raw_action_counts=raw_action_counts,
                    raw_action_metadata=raw_action_metadata,
                    mode="test",
                    max_batches=rate_args.rate_max_batches,
                )
                summary["dataset"] = _dataset_name(options)
                summary["rate_label"] = (
                    "greedy" if action_mode == "greedy" else f"K={top_k}" if top_k else "unrestricted"
                )
                summaries.append(summary)
                logger.info(
                    "Rate result %s: Hits@1=%s MRR=%s PED=%s",
                    summary["rate_label"],
                    summary["hits_at_1"],
                    summary["mrr"],
                    summary["ped"],
                )

        greedy = next(summary for summary in summaries if summary["mode"] == "greedy")
        unrestricted = next(
            (summary for summary in summaries if summary["mode"] == "tf_policy"), None
        )
        comparison = None
        if unrestricted is not None:
            comparison = {
                "hits_at_1_delta_greedy_minus_unrestricted": (
                    greedy["hits_at_1"] - unrestricted["hits_at_1"]
                ),
                "mrr_delta_greedy_minus_unrestricted": greedy["mrr"] - unrestricted["mrr"],
                "matches_hits_at_1_within_1e-6": bool(
                    abs(greedy["hits_at_1"] - unrestricted["hits_at_1"]) <= 1e-6
                ),
                "note": "No domain-specific threshold for 'nearly unchanged' was assumed.",
            }
            if comparison["matches_hits_at_1_within_1e-6"]:
                logger.warning(
                    "Greedy matches unrestricted Hits@1 within 1e-6; this is a scientific result."
                )

        output_dir = os.path.join(options["output_dir"], "rate_sweep")
        metadata = {
            "dataset": _dataset_name(options),
            "rate_arguments": vars(rate_args),
            "minerva_options": options,
            "checkpoint": _checkpoint_identity(options["model_load_dir"]),
            "root_git_head": _git_revision(os.getcwd()),
            "minerva_git_head": _git_revision(os.path.join(os.getcwd(), "minerva")),
            "task_agnostic_prior": prior_metadata,
            "truncation_diagnostics": raw_action_metadata,
            "greedy_unrestricted_comparison": comparison,
            "scientific_notes": {
                "greedy_zero_payload": "Requires synchronized policy, state/task, action set, and ordering.",
                "topk_budget": "A hard retained-support/local-rank bound, not entropy-coded traffic.",
                "sampling_backend": (
                    "Top-K uses seeded NumPy PCG64; unrestricted tf_policy uses upstream "
                    "TensorFlow categorical sampling. Equal distributions can have different "
                    "finite-sample metrics across those backends."
                ),
                "gold_hop_cost": "A diagnostic mask over the unchanged fixed-horizon trajectory.",
                "ped": "Only entity-edge path edit distance is labeled PED; relation edit distance is separate.",
            },
        }
        paths = save_rate_sweep_outputs(summaries, metadata, output_dir)
        paths.update(
            generate_rate_sweep_plots(
                summaries, output_dir, title_prefix=_dataset_name(options).upper()
            )
        )
        for name, path in paths.items():
            if path is not None:
                logger.info("Saved %s to %s", name, path)
    finally:
        embedding_server.close()
        if trainer is not None:
            trainer.close()


if __name__ == "__main__":
    main()
