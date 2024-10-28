import onnxruntime as ort
import torch
import numpy as np
import time

from cotracker.core.cotracker3 import CoTrackerThreeOnline

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ort_session_encoder = ort.InferenceSession("./onnx_models/fnet.onnx")
ort_session_mlp = ort.InferenceSession("./onnx_models/corr_mlp.onnx")
ort_session_transformer = ort.InferenceSession("./onnx_models/updateformer.onnx")

def to_numpy(tensor):
    tensor = tensor.detach().numpy() if tensor.requires_grad else tensor.numpy()
    tensor = tensor.cpu() if tensor.device == 'cuda' else tensor
    return tensor

dummy_input_fnet = torch.randn(16, 3, 384, 512)
dummy_input_updateformer = torch.randn(1, 100, 16, 1110)
dummy_input_corr_mlp = torch.randn(1600, 2401)

dummy_input_encoder_np = to_numpy(dummy_input_fnet)
start_time = time.time()
outputs_encoder = ort_session_encoder.run(None, {'input': dummy_input_encoder_np})
print(f'encoder_time: {time.time() - start_time}')
outputs_encoder = np.array(outputs_encoder)
outputs_encoder = outputs_encoder.squeeze(0)

dummy_input_transformer_np = to_numpy(dummy_input_updateformer)
start_time = time.time()
outputs_transformer = ort_session_transformer.run(None, {'input': dummy_input_transformer_np})
print(f'transformer_time: {time.time() - start_time}')
outputs_transformer = np.array(outputs_transformer)
outputs_transformer = outputs_transformer.squeeze(0)

dummy_input_mlp_np = to_numpy(dummy_input_corr_mlp)
start_time = time.time()
outputs_mlp = ort_session_mlp.run(None, {'input': dummy_input_mlp_np})
print(f'mlp_time: {time.time() - start_time}')
outputs_mlp = np.array(outputs_mlp)
outputs_mlp = outputs_mlp.squeeze(0)

print(f'outputs_encoder: {outputs_encoder.shape}')
print(f'outputs_transformer: {outputs_transformer.shape}')
print(f'outputs_mlp: {outputs_mlp.shape}')

pt_model = CoTrackerThreeOnline(window_len=16)
pt_model.load_state_dict(torch.load('./checkpoints/baseline_online.pth', map_location=torch.device('cpu')))

start_time = time.time()
pt_outputs_encoder = pt_model.fnet(dummy_input_fnet)
print(f'pt_encoder_time: {time.time() - start_time}')
start_time = time.time()
pt_outputs_transformer = pt_model.updateformer(dummy_input_updateformer)
print(f'pt_transformer_time: {time.time() - start_time}')
start_time = time.time()
pt_outputs_mlp = pt_model.corr_mlp(dummy_input_corr_mlp)
print(f'pt_mlp_time: {time.time() - start_time}')

print(f'pt_outputs_encoder: {pt_outputs_encoder.shape}')
print(f'pt_outputs_transformer: {pt_outputs_transformer.shape}')
print(f'pt_outputs_mlp: {pt_outputs_mlp.shape}')

np.testing.assert_allclose(outputs_encoder, to_numpy(pt_outputs_encoder), rtol=1e-02, atol=1e-04)
np.testing.assert_allclose(outputs_transformer, to_numpy(pt_outputs_transformer), rtol=1e-02, atol=1e-04)
np.testing.assert_allclose(outputs_mlp, to_numpy(pt_outputs_mlp), rtol=1e-02, atol=1e-04)