import tensorrt as trt
import torch
from cuda import cudart
import ctypes
# from pytorch_lightning import seed_everything


def compute_fmha(qkv_input):
    q = qkv_input[:, :, 0, :, :].squeeze()
    k = qkv_input[:, :, 1, :, :].squeeze()
    v = qkv_input[:, :, 2, :, :].squeeze()

    qkv_shape = q.shape
    print(qkv_shape)
    q = q.permute((0,2,1,3)).reshape(qkv_shape[0]* qkv_shape[2], qkv_shape[1], qkv_shape[3])
    k = k.permute((0,2,1,3)).reshape(qkv_shape[0]* qkv_shape[2], qkv_shape[1], qkv_shape[3])
    v = v.permute((0,2,1,3)).reshape(qkv_shape[0]* qkv_shape[2], qkv_shape[1], qkv_shape[3])

    k = k.permute((0,2,1))
    print(k.shape)

    qk = torch.matmul(q, k) * 0.15811388194561005

    softmax_out = torch.nn.functional.softmax(qk, dim=-1)

    output = torch.matmul(softmax_out, v)
    print(output.shape)

    output = output.reshape(qkv_shape[0], qkv_shape[2], qkv_shape[1], qkv_shape[3])
    
    output = output.permute((0,2,1,3))

    print(output.shape)
    return output



torch.manual_seed(18279)

plugin_lib = "/workdir/tmp/trt_fmha//build/out/libnvinfer_plugin.so"
ctypes.cdll.LoadLibrary(plugin_lib)

trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, '')

with open("fmha_v2.engine", "rb") as f:
    engine_str = f.read()
engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_str)
context = engine.create_execution_context()

# seed_everything(1892849)
qkv = torch.randn((2,1536,3,8,40), dtype=torch.float16).cuda().contiguous()

output_base = compute_fmha(qkv)

# print(output_base.reshape(-1)[:100])
# k = torch.randn((2,16384,8,40), dtype=torch.float16).cuda().contiguous()
# v = torch.randn((2,16384,8,40), dtype=torch.float16).cuda().contiguous()
output = torch.zeros((2,1536,8,40), dtype=torch.float16).cuda().contiguous()

tensor_list = [qkv, output]

for i in range(2):
    name = engine.get_tensor_name(i)
    context.set_tensor_address(name, tensor_list[i].data_ptr())
_, stream = cudart.cudaStreamCreate()
context.execute_async_v3(stream)

# print(output.reshape(-1)[:100])
print(output)
print(output_base)

max_diff = torch.max(torch.abs(output - output_base))
max_out = torch.max(torch.abs(output))
print("max diff: ", max_diff)
print("max output: ", max_out)
print("finished")
