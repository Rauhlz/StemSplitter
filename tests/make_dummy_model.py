# Builds a dummy ONNX model with the same I/O signature as the exported HTDemucs core:
#   inputs : mix [1,2,T]  spec [1,4,2048,F]
#   outputs: spec_out [1,S,4,2048,F]  wave_out [1,S,2,T]
# spec_out[s] = spec * ws[s]; wave_out[s] = mix * wt[s]
import sys, numpy as np
try:
    from onnx import onnx_pb as P          # if the onnx package is installed
except ImportError:                        # otherwise the bundled generated protobuf module
    sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), 'proto'))
    import onnx_pb2 as P

S = int(sys.argv[2]) if len(sys.argv) > 2 else 4
# stem 0 = frequency branch only, stem 1 = time branch only, stem 2 = a mix of both, rest = silence
ws = np.zeros(S, np.float32); ws[0] = 1; ws[2] = 0.5
wt = np.zeros(S, np.float32); wt[1] = 1; wt[2] = -0.25
T, F = 343980, 336
m = P.ModelProto(); m.ir_version = 8; m.producer_name = "dummy"
op = m.opset_import.add(); op.domain = ""; op.version = 17
g = m.graph; g.name = "dummy"

def vi(coll, name, shape):
    v = coll.add(); v.name = name
    tt = v.type.tensor_type; tt.elem_type = 1
    for d in shape: tt.shape.dim.add().dim_value = d

def init(name, arr):
    t = g.initializer.add(); t.name = name; t.data_type = 1
    t.dims.extend(arr.shape); t.raw_data = arr.astype(np.float32).tobytes()

def node(op_type, ins, outs):
    n = g.node.add(); n.op_type = op_type; n.input.extend(ins); n.output.extend(outs)

vi(g.input, "mix", [1, 2, T]); vi(g.input, "spec", [1, 4, 2048, F])
init("ws", ws.reshape(1, S, 1, 1, 1)); init("wt", wt.reshape(1, S, 1, 1))
t = g.initializer.add(); t.name = "ax1"; t.data_type = 7; t.dims.append(1); t.raw_data = np.array([1], np.int64).tobytes()
node("Unsqueeze", ["spec", "ax1"], ["spec_u"]); node("Mul", ["spec_u", "ws"], ["spec_out"])
node("Unsqueeze", ["mix", "ax1"], ["mix_u"]);  node("Mul", ["mix_u", "wt"], ["wave_out"])
vi(g.output, "spec_out", [1, S, 4, 2048, F]); vi(g.output, "wave_out", [1, S, 2, T])
if S == 6:
    md = m.metadata_props.add(); md.key = "sources"; md.value = "drums,bass,other,vocals,guitar,piano"
open(sys.argv[1], "wb").write(m.SerializeToString())
print("wrote", sys.argv[1])
