#!/usr/bin/env python3
"""
Compare two ONNX files to verify export consistency.

What it checks:
  1) File hash (raw bytes) + model hash (normalized, metadata-insensitive)
  2) Graph structure (inputs/outputs/value_info, nodes, attrs)
  3) Weights/initializers (by name + tensor content)
  4) Optional: runtime numerical check via onnxruntime (if installed)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _try_import_onnx():
    try:
        import onnx  # type: ignore

        return onnx
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: onnx. Install with: pip install onnx"
        ) from e


def _tensor_to_numpy(initializer) -> "Any":
    onnx = _try_import_onnx()
    from onnx import numpy_helper  # type: ignore

    return numpy_helper.to_array(initializer)


def _normalize_model_for_hash(model, ignore_metadata: bool) -> bytes:
    """
    Return a deterministic serialization suitable for hashing.
    If ignore_metadata=True, strip fields that commonly change between exports.
    """
    m = copy.deepcopy(model)
    if ignore_metadata:
        # Do NOT modify ir_version: checker requires it to be valid.
        m.producer_name = ""
        m.producer_version = ""
        m.domain = ""
        m.model_version = 0
        m.doc_string = ""
        del m.metadata_props[:]
        if m.HasField("graph"):
            m.graph.doc_string = ""
        for opset in m.opset_import:
            opset.version = int(opset.version)  # keep numeric, but stable

    return m.SerializeToString(deterministic=True)


@dataclass
class CompareResult:
    equal: bool
    summary: str
    details: List[str]


def _fmt_vi(vi) -> Tuple[str, str, Tuple[int, ...] | Tuple[str, ...]]:
    """
    Represent a ValueInfoProto (name, elem_type, shape).
    Shape uses ints when available, otherwise dim_param strings.
    """
    onnx = _try_import_onnx()
    from onnx import TensorProto  # type: ignore

    name = vi.name
    t = vi.type
    if not t.HasField("tensor_type"):
        return (name, "non_tensor", ())
    elem = t.tensor_type.elem_type
    elem_name = TensorProto.DataType.Name(elem) if elem in TensorProto.DataType.values() else str(elem)
    shape = []
    if t.tensor_type.HasField("shape"):
        for d in t.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(int(d.dim_value))
            elif d.HasField("dim_param"):
                shape.append(str(d.dim_param))
            else:
                shape.append("?")
    return (name, elem_name, tuple(shape))


def _attr_to_py(attr) -> Any:
    onnx = _try_import_onnx()
    from onnx import AttributeProto, numpy_helper  # type: ignore

    t = attr.type
    if t == AttributeProto.FLOAT:
        return float(attr.f)
    if t == AttributeProto.INT:
        return int(attr.i)
    if t == AttributeProto.STRING:
        return bytes(attr.s)
    if t == AttributeProto.FLOATS:
        return tuple(float(x) for x in attr.floats)
    if t == AttributeProto.INTS:
        return tuple(int(x) for x in attr.ints)
    if t == AttributeProto.STRINGS:
        return tuple(bytes(x) for x in attr.strings)
    if t == AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t).tobytes()
    if t == AttributeProto.TENSORS:
        return tuple(numpy_helper.to_array(t0).tobytes() for t0 in attr.tensors)
    if t == AttributeProto.GRAPH:
        return attr.g.SerializeToString(deterministic=True)
    if t == AttributeProto.GRAPHS:
        return tuple(g.SerializeToString(deterministic=True) for g in attr.graphs)
    if t == AttributeProto.SPARSE_TENSOR:
        return attr.sparse_tensor.SerializeToString(deterministic=True)
    if t == AttributeProto.SPARSE_TENSORS:
        return tuple(st.SerializeToString(deterministic=True) for st in attr.sparse_tensors)
    return attr.SerializeToString(deterministic=True)


def _node_fingerprint(node) -> Tuple[Any, ...]:
    attrs = sorted(((a.name, a.type, _attr_to_py(a)) for a in node.attribute), key=lambda x: x[0])
    return (
        node.op_type,
        node.domain,
        tuple(node.input),
        tuple(node.output),
        attrs,
    )


def _compare_initializers(g1, g2) -> Tuple[bool, List[str]]:
    inits1 = {t.name: t for t in g1.initializer}
    inits2 = {t.name: t for t in g2.initializer}
    details: List[str] = []

    if set(inits1.keys()) != set(inits2.keys()):
        missing = sorted(set(inits1.keys()) - set(inits2.keys()))
        extra = sorted(set(inits2.keys()) - set(inits1.keys()))
        if missing:
            details.append(f"Initializer missing in model2: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if extra:
            details.append(f"Initializer extra in model2: {extra[:10]}{' ...' if len(extra) > 10 else ''}")
        return False, details

    for name in sorted(inits1.keys()):
        t1 = inits1[name]
        t2 = inits2[name]
        if (t1.data_type, tuple(t1.dims)) != (t2.data_type, tuple(t2.dims)):
            details.append(
                f"Initializer shape/type differs for '{name}': "
                f"{(t1.data_type, tuple(t1.dims))} vs {(t2.data_type, tuple(t2.dims))}"
            )
            return False, details

        # Compare content via numpy bytes hash (fast and deterministic)
        a1 = _tensor_to_numpy(t1)
        a2 = _tensor_to_numpy(t2)
        if a1.dtype != a2.dtype or a1.shape != a2.shape:
            details.append(f"Initializer array differs for '{name}': {a1.dtype}{a1.shape} vs {a2.dtype}{a2.shape}")
            return False, details
        h1 = _sha256_bytes(a1.tobytes())
        h2 = _sha256_bytes(a2.tobytes())
        if h1 != h2:
            details.append(f"Initializer content differs for '{name}' (sha256 {h1[:12]}.. vs {h2[:12]}..)")
            return False, details

    return True, details


def compare_onnx(
    path1: str,
    path2: str,
    *,
    ignore_metadata: bool = True,
    check_runtime: bool = False,
    seed: int = 0,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    dynamic_dim_default: int = 1,
) -> CompareResult:
    onnx = _try_import_onnx()

    b1 = _read_file_bytes(path1)
    b2 = _read_file_bytes(path2)
    file_hash1 = _sha256_bytes(b1)
    file_hash2 = _sha256_bytes(b2)

    m1 = onnx.load_from_string(b1)
    m2 = onnx.load_from_string(b2)

    onnx.checker.check_model(m1)
    onnx.checker.check_model(m2)

    norm_hash1 = _sha256_bytes(_normalize_model_for_hash(m1, ignore_metadata=ignore_metadata))
    norm_hash2 = _sha256_bytes(_normalize_model_for_hash(m2, ignore_metadata=ignore_metadata))

    details: List[str] = []
    details.append(f"file_sha256_1={file_hash1}")
    details.append(f"file_sha256_2={file_hash2}")
    details.append(f"norm_sha256_1={norm_hash1} (ignore_metadata={ignore_metadata})")
    details.append(f"norm_sha256_2={norm_hash2} (ignore_metadata={ignore_metadata})")

    g1, g2 = m1.graph, m2.graph

    # Inputs/outputs/value_info
    ins1 = [_fmt_vi(x) for x in g1.input]
    ins2 = [_fmt_vi(x) for x in g2.input]
    outs1 = [_fmt_vi(x) for x in g1.output]
    outs2 = [_fmt_vi(x) for x in g2.output]
    if ins1 != ins2:
        return CompareResult(False, "Graph inputs differ", details + [f"inputs_1={ins1}", f"inputs_2={ins2}"])
    if outs1 != outs2:
        return CompareResult(False, "Graph outputs differ", details + [f"outputs_1={outs1}", f"outputs_2={outs2}"])

    # Initializers
    ok_init, init_details = _compare_initializers(g1, g2)
    if not ok_init:
        return CompareResult(False, "Initializers differ", details + init_details)

    # Nodes
    nodes1 = [_node_fingerprint(n) for n in g1.node]
    nodes2 = [_node_fingerprint(n) for n in g2.node]
    if len(nodes1) != len(nodes2):
        return CompareResult(
            False,
            f"Node count differs: {len(nodes1)} vs {len(nodes2)}",
            details,
        )
    for i, (n1, n2) in enumerate(zip(nodes1, nodes2)):
        if n1 != n2:
            return CompareResult(False, f"Node differs at index {i}", details + [f"node1={n1}", f"node2={n2}"])

    # If normalized hashes match, treat as equal (strong signal)
    if norm_hash1 == norm_hash2:
        if check_runtime:
            rt_ok, rt_details = _runtime_compare(
                m1,
                m2,
                seed=seed,
                rtol=rtol,
                atol=atol,
                dynamic_dim_default=dynamic_dim_default,
            )
            if not rt_ok:
                return CompareResult(False, "Runtime outputs differ", details + rt_details)
        return CompareResult(True, "Models match", details)

    # If graph + init + nodes match but hash differs, likely metadata or ordering elsewhere.
    # Provide a conservative mismatch.
    if check_runtime:
        rt_ok, rt_details = _runtime_compare(
            m1,
            m2,
            seed=seed,
            rtol=rtol,
            atol=atol,
            dynamic_dim_default=dynamic_dim_default,
        )
        if rt_ok:
            return CompareResult(
                True,
                "Structurally identical; runtime outputs match (hash differs)",
                details + ["Note: normalized hash differs; likely non-essential fields differ."],
            )
        return CompareResult(False, "Hash differs and runtime outputs differ", details + rt_details)

    return CompareResult(False, "Normalized hash differs (metadata/order may differ)", details)


def _pick_dim(dim, default: int) -> int:
    if dim is None:
        return default
    if isinstance(dim, int):
        return dim if dim > 0 else default
    return default


def _make_random_input(vi, rng, dynamic_dim_default: int):
    import numpy as np

    t = vi.type.tensor_type
    elem = t.elem_type
    shape = []
    for d in t.shape.dim:
        if d.HasField("dim_value") and int(d.dim_value) > 0:
            shape.append(int(d.dim_value))
        else:
            shape.append(dynamic_dim_default)

    # Common types only; fallback to float32
    if elem in (1, 10, 11):  # FLOAT, FLOAT16, DOUBLE
        dtype = np.float32
        return rng.standard_normal(size=shape, dtype=dtype)
    if elem in (7,):  # INT64
        return rng.integers(low=0, high=10, size=shape, dtype=np.int64)
    if elem in (6,):  # INT32
        return rng.integers(low=0, high=10, size=shape, dtype=np.int32)
    if elem in (9,):  # BOOL
        return rng.integers(low=0, high=2, size=shape, dtype=np.int8).astype(np.bool_)
    # UINT8 / others
    return rng.standard_normal(size=shape).astype(np.float32)


def _runtime_compare(
    m1,
    m2,
    *,
    seed: int,
    rtol: float,
    atol: float,
    dynamic_dim_default: int,
) -> Tuple[bool, List[str]]:
    details: List[str] = []
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        return False, [f"onnxruntime not available: {e}. Install with: pip install onnxruntime"]

    import numpy as np

    so = ort.SessionOptions()
    so.log_severity_level = 3

    s1 = ort.InferenceSession(m1.SerializeToString(deterministic=True), sess_options=so, providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(m2.SerializeToString(deterministic=True), sess_options=so, providers=["CPUExecutionProvider"])

    ins1 = s1.get_inputs()
    ins2 = s2.get_inputs()
    names1 = [i.name for i in ins1]
    names2 = [i.name for i in ins2]
    if names1 != names2:
        return False, [f"runtime input name list differs: {names1} vs {names2}"]

    rng = np.random.default_rng(seed)
    feeds: Dict[str, Any] = {}
    # Use model (protobuf) ValueInfo to infer element types/shapes if possible
    vi_map = {vi.name: vi for vi in m1.graph.input}
    for inp in ins1:
        vi = vi_map.get(inp.name)
        if vi is None or not vi.type.HasField("tensor_type"):
            # fallback: use ort input shape/type
            shape = [dynamic_dim_default if (d is None or int(d) <= 0) else int(d) for d in (inp.shape or [])]
            feeds[inp.name] = rng.standard_normal(size=shape).astype(np.float32)
        else:
            feeds[inp.name] = _make_random_input(vi, rng, dynamic_dim_default=dynamic_dim_default)

    out1 = s1.run(None, feeds)
    out2 = s2.run(None, feeds)

    if len(out1) != len(out2):
        return False, [f"runtime output count differs: {len(out1)} vs {len(out2)}"]

    for i, (a, b) in enumerate(zip(out1, out2)):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            return False, [f"output[{i}] shape differs: {a.shape} vs {b.shape}"]
        if a.dtype != b.dtype:
            details.append(f"output[{i}] dtype differs: {a.dtype} vs {b.dtype} (comparing as float32)")
            a = a.astype(np.float32)
            b = b.astype(np.float32)
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            diff = np.max(np.abs(a - b)) if a.size else 0.0
            return False, [f"output[{i}] not close (rtol={rtol}, atol={atol}), max_abs_diff={diff}"]

    return True, details + [f"runtime check passed (seed={seed}, rtol={rtol}, atol={atol})"]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare two ONNX models for equivalence.")
    p.add_argument("onnx1", type=str, help="Path to first ONNX file")
    p.add_argument("onnx2", type=str, help="Path to second ONNX file")
    p.add_argument(
        "--strict-metadata",
        action="store_true",
        help="Also compare metadata fields (by default metadata is ignored for hashing).",
    )
    p.add_argument(
        "--runtime",
        action="store_true",
        help="Also run a numerical comparison using onnxruntime (if installed).",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for runtime inputs (default: 0)")
    p.add_argument("--rtol", type=float, default=1e-3, help="rtol for runtime output comparison")
    p.add_argument("--atol", type=float, default=1e-4, help="atol for runtime output comparison")
    p.add_argument(
        "--dynamic-dim-default",
        type=int,
        default=1,
        help="Default size for dynamic/unknown dimensions during runtime check",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    res = compare_onnx(
        args.onnx1,
        args.onnx2,
        ignore_metadata=not args.strict_metadata,
        check_runtime=args.runtime,
        seed=args.seed,
        rtol=args.rtol,
        atol=args.atol,
        dynamic_dim_default=args.dynamic_dim_default,
    )

    if res.equal:
        print("EQUAL: " + res.summary)
        for d in res.details:
            print("  " + d)
        return 0

    print("NOT EQUAL: " + res.summary)
    for d in res.details:
        print("  " + d)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

