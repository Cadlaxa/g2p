# coding=utf-8
import os
import datetime
import re
import hydra
import numpy as np
from collections import Counter
from omegaconf import OmegaConf
import onnx
import onnxruntime as ort
import torch
from torch import nn
import editdistance
from dataset import BOS_IDX, EOS_IDX


class G2pEncode(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src):
        src = torch.cat(
            [
                torch.tensor([[BOS_IDX]], dtype=torch.int32),
                src,
                torch.tensor([[EOS_IDX]], dtype=torch.int32),
            ],
            dim=1,
        )
        tgt, mem, hc, ts = self.model.encode(src)
        return tgt, mem, hc, ts


class G2pDecodeStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, cond, tgt_in, mem, src_lengths, hc_in, ts_in):
        tgt_out, hc_out, ts_out = self.model.decode_step(
            tgt_in, mem, src_lengths, hc_in, ts_in
        )
        return cond, tgt_out, mem, src_lengths, hc_out, ts_out


def get_edit_operations(gold: list, pred: list) -> dict:
    from difflib import SequenceMatcher

    operations = {
        "ins": [],
        "del": [],
        "sub": [],
    }

    matcher = SequenceMatcher(None, gold, pred)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pass
        elif tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                if i1 + k < i2 and j1 + k < j2:
                    operations["sub"].append(
                        {
                            "gold": gold[i1 + k],
                            "pred": pred[j1 + k],
                        }
                    )
                elif i1 + k < i2:
                    operations["del"].append({"phoneme": gold[i1 + k]})
                else:
                    operations["ins"].append({"phoneme": pred[j1 + k]})
        elif tag == "insert":
            for k in range(j1, j2):
                operations["ins"].append({"phoneme": pred[k]})
        elif tag == "delete":
            for k in range(i1, i2):
                operations["del"].append({"phoneme": gold[k]})
    return operations


def evaluate_validation_set(ckpt_path: str, training_module, datamodule):
    output_log = ckpt_path.replace(".ckpt", "_val_analysis.log")

    datamodule.setup(stage="fit")
    val_loader = datamodule.val_dataloader()

    wrong_words = []
    substitution_pairs = Counter()
    insertion_phonemes = Counter()
    deletion_phonemes = Counter()

    total_words = 0
    total_word_errors = 0
    total_phoneme_errors = 0
    total_phoneme_count = 0

    print("Evaluating on validation set...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            src, tgt, src_lengths, tgt_lengths = batch

            tgt_out = training_module.model.greedy_decode(
                src, src_lengths, return_early=True
            )

            total_words += src.size(0)

            for i in range(src.size(0)):
                word = [
                    cfg.symbols.graphemes[idx] for idx in src[i].tolist() if idx >= 4
                ]

                gold = [
                    cfg.symbols.phonemes[idx] for idx in tgt[i].tolist() if idx >= 4
                ]
                # gold = [re.sub(r"\d+$", "", p) for p in gold]
                pred = [
                    cfg.symbols.phonemes[idx]
                    for idx in tgt_out[i][1:].tolist()
                    if idx >= 4
                ]
                # pred = [re.sub(r"\d+$", "", p) for p in pred]

                dis = editdistance.distance(gold, pred)
                total_phoneme_count += len(gold)
                if dis > 0:
                    total_word_errors += 1
                    total_phoneme_errors += dis

                    operations = get_edit_operations(gold, pred)

                    wrong_words.append(
                        {
                            "word": "".join(word),
                            "gold": "-".join(gold),
                            "pred": "-".join(pred),
                            "distance": dis,
                            "operations": operations,
                        }
                    )

                    for sub in operations["sub"]:
                        substitution_pairs[(sub["gold"], sub["pred"])] += 1
                    for ins in operations["ins"]:
                        insertion_phonemes[ins["phoneme"]] += 1
                    for dels in operations["del"]:
                        deletion_phonemes[dels["phoneme"]] += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches")

    with open(output_log, "w", encoding="utf-8") as f:
        f.write("=== G2P Validation Error Analysis ===\n\n")

        wer = (total_word_errors / total_words * 100) if total_words > 0 else 0
        per = (
            (total_phoneme_errors / total_phoneme_count * 100)
            if total_phoneme_count > 0
            else 0
        )

        f.write(f"Overall Statistics:\n")
        f.write(f"  Total words: {total_words}\n")
        f.write(f"  Word errors: {total_word_errors}\n")
        f.write(f"  WER: {wer:.2f}%\n")
        f.write(f"  Total phonemes: {total_phoneme_count}\n")
        f.write(f"  Phoneme errors: {total_phoneme_errors}\n")
        f.write(f"  PER: {per:.2f}%\n\n")

        f.write(f"Error Operation Statistics:\n")
        total_subs = sum(len(w["operations"]["sub"]) for w in wrong_words)
        total_ins = sum(len(w["operations"]["ins"]) for w in wrong_words)
        total_dels = sum(len(w["operations"]["del"]) for w in wrong_words)
        f.write(
            f"  Total sub: {total_subs} ({total_subs / total_phoneme_count * 100:.2f}%)\n"
        )
        f.write(
            f"  Total ins: {total_ins} ({total_ins / total_phoneme_count * 100:.2f}%)\n"
        )
        f.write(
            f"  Total del: {total_dels} ({total_dels / total_phoneme_count * 100:.2f}%)\n\n"
        )

        if substitution_pairs:
            f.write(f"Top 20 Substitution Pairs (gold -> pred):\n")
            for (gold, pred), count in substitution_pairs.most_common(20):
                f.write(f"  {gold:12s} -> {pred:12s}: {count:3d} times\n")
            f.write("\n")

        if insertion_phonemes:
            f.write(f"Top 20 Inserted Phonemes:\n")
            for phoneme, count in insertion_phonemes.most_common(20):
                f.write(f"  {phoneme:12s}: {count:3d} times\n")
            f.write("\n")

        if deletion_phonemes:
            f.write(f"Top 20 Deleted Phonemes:\n")
            for phoneme, count in deletion_phonemes.most_common(20):
                f.write(f"  {phoneme:12s}: {count:3d} times\n")
            f.write("\n")

        f.write(f"Wrong Words ({len(wrong_words)} total):\n")
        f.write("-" * 80 + "\n")
        for idx, error in enumerate(
            sorted(wrong_words, key=lambda x: x["distance"], reverse=True)[:500]
        ):
            f.write(
                f"{idx+1:3d}. {error['word']:15s} | Gold: {error['gold']:30s} | Pred: {error['pred']:30s} | Distance: {error['distance']}\n"
            )

    print(f"\nValidation Results:")
    print(f"  WER: {wer:.2f}%")
    print(f"  PER: {per:.2f}%")
    print(f"  Error log written to: {output_log}")

    return training_module.model


def export(cfg, ckpt_path: str, training_module, datamodule):
    """Export G2p model to ONNX format."""
    onnx_encode_path = ckpt_path.replace(".ckpt", "_encode.onnx")
    onnx_decode_path = ckpt_path.replace(".ckpt", "_decode_step.onnx")
    onnx_full_path = ckpt_path.replace(".ckpt", ".onnx")

    dataset = datamodule.dataset
    model = training_module.model
    g2p_encode = G2pEncode(model)
    g2p_decode_step = G2pDecodeStep(model)

    src, _ = dataset[999]
    src = src.unsqueeze(0)

    with torch.no_grad():
        g2p_encode.eval()
        torch.onnx.export(
            g2p_encode,
            (src[:, 1:-1]),
            onnx_encode_path,
            input_names=["src"],
            output_names=["tgt", "mem", "hc", "ts"],
            dynamic_axes={
                "src": {1: "T"},
                "mem": {1: "T"},
            },
            opset_version=20,
        )

        src_lengths = torch.tensor([src.size(1)]).int()
        tgt, mem, hc, ts = g2p_encode(src[:, 1:-1])
        cond = torch.tensor(True, dtype=torch.bool)
        g2p_decode_step.eval()
        torch.onnx.export(
            g2p_decode_step,
            (cond, tgt, mem, src_lengths, hc, ts),
            onnx_decode_path,
            input_names=[
                "cond",
                "tgt",
                "mem",
                "src_lengths",
                "hc",
                "ts",
            ],
            output_names=[
                "cond_out",
                "tgt_out",
                "mem_out",
                "src_lengths_out",
                "hc_out",
                "ts_out",
            ],
            dynamic_axes={
                "mem": {1: "T"},
                "mem_out": {1: "T"},
            },
            opset_version=20,
        )

    # test onnx inference
    ort_encode = ort.InferenceSession(onnx_encode_path)
    ort_decode_step = ort.InferenceSession(onnx_decode_path)
    ort_inputs = {"src": src[:, 1:-1].numpy()}
    ort_outs = ort_encode.run(None, ort_inputs)
    tgt, mem, hc, ts = ort_outs
    cond = np.array(True, dtype=bool)
    src_lengths = np.array([src.size(1)]).astype(np.int32)
    for i in range(model.max_len):
        ort_inputs = {
            "cond": cond,
            "tgt": tgt,
            "mem": mem,
            "src_lengths": src_lengths,
            "hc": hc,
            "ts": ts,
        }
        ort_outs = ort_decode_step.run(None, ort_inputs)
        cond, tgt, mem, src_lengths, hc, ts = ort_outs
    word = [cfg.symbols.graphemes[i.item()] for i in src[0] if i > 3]
    phs = [cfg.symbols.phonemes[i.item()] for i in tgt[0] if i > 3]
    print("ONNX test result:", "".join(word), "-".join(phs))

    # merge graphs
    enc_model = onnx.load(onnx_encode_path)
    step_model = onnx.load(onnx_decode_path)
    main_model = merge_models(enc_model, step_model, model.max_len)
    onnx.save(main_model, onnx_full_path)
    print(f"Merged ONNX model saved to: {onnx_full_path}")
    os.remove(onnx_encode_path)
    os.remove(onnx_decode_path)

    # test merged model
    ort_session = ort.InferenceSession(onnx_full_path)
    ort_inputs = {
        "src": src[:, 1:-1].numpy(),
    }
    start_time = datetime.datetime.now()
    ort_outs = ort_session.run(None, ort_inputs)
    total_time = datetime.datetime.now() - start_time
    tgt = ort_outs[0]
    print(f"ONNX seq2seq total time: {total_time.total_seconds()*1000:.2f} ms")
    phs = [cfg.symbols.phonemes[i.item()] for i in tgt[0] if i > 3]
    print("ONNX seq2seq test result:", "".join(word), "-".join(phs))

    print("testing single graphemes:")
    for i in range(4, len(cfg.symbols.graphemes)):
        ort_inputs = {
            "src": np.array([[i]], dtype=np.int32),
        }
        ort_outs = ort_session.run(None, ort_inputs)
        tgt = ort_outs[0]
        phs = [cfg.symbols.phonemes[i.item()] for i in tgt[0] if i > 3]
        print(f"  {cfg.symbols.graphemes[i]} -> {'-'.join(phs)}")

    if cfg.export.oudep.id:
        pack_oudep(cfg, dataset, onnx_full_path)


def merge_models(enc_model, step_model, max_len):

    def add_prefix_to_graph(graph, prefix):
        def add_prefix(name):
            if not name:
                return name
            if name.startswith("model.") and (
                name.endswith(".weight") or name.endswith(".bias")
            ):
                return f"{prefix}.{name}"
            if name.startswith("/"):
                return f"/{prefix}{name}"
            return f"{prefix}_{name}"

        for node in graph.node:
            node.name = add_prefix(node.name)
            node.input[:] = [add_prefix(i) for i in node.input]
            node.output[:] = [add_prefix(o) for o in node.output]
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    add_prefix_to_graph(attr.g, prefix)
        for init in graph.initializer:
            init.name = add_prefix(init.name)
        for val in graph.value_info:
            val.name = add_prefix(val.name)
        for inp in graph.input:
            inp.name = add_prefix(inp.name)
        for out in graph.output:
            out.name = add_prefix(out.name)

    def connect(input, output):
        return onnx.helper.make_node("Identity", inputs=[input], outputs=[output])

    add_prefix_to_graph(enc_model.graph, "enc")
    add_prefix_to_graph(step_model.graph, "dec")

    loop_body_graph = onnx.GraphProto()
    loop_body_graph.CopyFrom(step_model.graph)
    loop_body_graph.name = "dec_loop_body"
    loop_body_graph.input.insert(
        0, onnx.helper.make_tensor_value_info("iter", onnx.TensorProto.INT64, [])
    )

    trip_count_tensor = onnx.helper.make_tensor(
        "trip_count", onnx.TensorProto.INT64, dims=[], vals=[max_len]
    )
    loop_cond = onnx.helper.make_tensor("loop_cond", onnx.TensorProto.BOOL, [], [True])
    loop_node = onnx.helper.make_node(
        "Loop",
        name="dec_loop",
        inputs=["trip_count", "loop_cond"]
        + [inp.name for inp in loop_body_graph.input[2:]],
        outputs=[out.name for out in loop_body_graph.output[1:]],
        body=loop_body_graph,
    )

    main_graph = onnx.GraphProto()
    main_graph.CopyFrom(enc_model.graph)
    main_graph.name = "g2p"
    main_graph.input.clear()
    main_graph.input.append(
        onnx.helper.make_tensor_value_info("src", onnx.TensorProto.INT32, [1, "T"])
    )
    main_graph.node.insert(0, connect("src", "enc_src"))

    def calc_src_lengths():
        # find output name concat node for src:
        for i, node in enumerate(main_graph.node):
            if node.op_type == "Concat" and "enc_src" in node.input:
                concat_src_output = node.output[0]
                break
        nodes = [
            onnx.helper.make_node(
                "Shape", inputs=[concat_src_output], outputs=["src_shape"]
            ),  # value=[B, T], shape=(2,), dtype=int64
            onnx.helper.make_node(
                "Gather",
                inputs=["src_shape", "src_len_idx"],
                outputs=["src_lengths"],
                axis=0,
            ),  # value=[T], shape=(1,), dtype=int64
            onnx.helper.make_node(
                "Unsqueeze",
                inputs=["src_lengths", "src_len_axes"],
                outputs=["src_lengths2"],
            ),  # value=[[T]], shape=(1, 1), dtype=int64
            onnx.helper.make_node(
                "Cast",
                inputs=["src_lengths2"],
                outputs=["dec_src_lengths"],
                to=onnx.TensorProto.INT32,
            ),  # value=[[T]], shape=(1, 1), dtype=int32
        ]
        initializers = [
            onnx.helper.make_tensor(
                "src_len_idx", onnx.TensorProto.INT64, dims=[], vals=[1]
            ),
            onnx.helper.make_tensor(
                "src_len_axes", onnx.TensorProto.INT64, dims=[1], vals=[0]
            ),
        ]
        return nodes, initializers

    src_len_nodes, src_len_initializers = calc_src_lengths()
    main_graph.node.extend(src_len_nodes)
    main_graph.initializer.extend(src_len_initializers)

    for inp, out in zip(
        ["enc_tgt", "enc_mem", "enc_hc", "enc_ts"],
        ["dec_tgt", "dec_mem", "dec_hc", "dec_ts"],
    ):
        main_graph.node.append(connect(inp, out))
    main_graph.node.extend([loop_node])
    main_graph.initializer.extend([trip_count_tensor, loop_cond])

    main_graph.node.append(connect("dec_tgt_out", "tgt"))
    main_graph.output.clear()
    main_graph.output.append(
        onnx.helper.make_tensor_value_info("tgt", onnx.TensorProto.INT32, [1, max_len])
    )
    main_model = onnx.helper.make_model(
        main_graph,
        opset_imports=[onnx.helper.make_operatorsetid("", 20)],
        ir_version=enc_model.ir_version,
    )
    onnx.checker.check_model(main_model)
    return main_model


def pack_oudep(cfg, dataset, onnx_full_path: str):
    import zipfile
    import yaml

    oudep_id = cfg.export.oudep.id
    oudep_ver = cfg.export.oudep.version
    output_path = os.path.join(
        os.path.dirname(onnx_full_path), f"{oudep_id}_v{oudep_ver}.oudep"
    )
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_LZMA) as zf:
        zf.write(onnx_full_path, arcname="model.onnx")
        zf.writestr(
            "oudep.yaml",
            yaml.dump(OmegaConf.to_object(cfg.export.oudep), allow_unicode=True),
        )
        zf.writestr(
            "config.yaml",
            yaml.dump(OmegaConf.to_object(cfg.export.config), allow_unicode=True),
        )
        print("Writing dictionary...")
        date_time = datetime.datetime.now().timetuple()[:6]
        zip_info = zipfile.ZipInfo("dict.txt", date_time=date_time)
        with zf.open(zip_info, "w") as dict_io:
            for w, phs in dataset:
                word = "".join(
                    [cfg.symbols.graphemes[idx] for idx in w.tolist() if idx >= 4]
                )
                phones = " ".join(
                    [cfg.symbols.phonemes[idx] for idx in phs.tolist() if idx >= 4]
                )
                dict_io.write(f"{word}  {phones}\n".encode("utf-8"))

    print(f"Packed oudep at: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Path to the model checkpoint (.ckpt)")
    args = parser.parse_args()

    ckpt_path = args.ckpt

    from g2p_module import G2pModule
    from data_module import G2pDataModule

    print(f"Loading checkpoint from: {args.ckpt}")
    hydra_dir = os.path.join(os.path.dirname(args.ckpt), "..", ".hydra")
    with hydra.initialize(config_path=hydra_dir, version_base=None):
        cfg = hydra.compose(config_name="config")

    dataset = hydra.utils.instantiate(cfg.dataset)
    datamodule = G2pDataModule(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        eval_split=cfg.train.eval_split,
        seed=cfg.train.get("seed", 42),
    )

    training_module = G2pModule.load_from_checkpoint(ckpt_path, weights_only=False)
    training_module.eval()
    training_module.to("cpu")

    evaluate_validation_set(ckpt_path, training_module, datamodule)
    export(cfg, ckpt_path, training_module, datamodule)
