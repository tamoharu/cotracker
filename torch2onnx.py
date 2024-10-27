import torch
import torch.onnx
from cotracker.core.cotracker3 import CoTrackerThreeOnline

window_len = 16
model_resolution = (384, 512)
model = CoTrackerThreeOnline(window_len=window_len, model_resolution=model_resolution)
model.load_state_dict(torch.load('./checkpoints/baseline_online.pth', map_location=torch.device('cpu')))

fnet = model.fnet
updateformer = model.updateformer
corr_mlp = model.corr_mlp

dummy_input_fnet = torch.randn(16, 3, 384, 512)
dummy_input_updateformer = torch.randn(1, 100, 16, 1110)
dummy_input_corr_mlp = torch.randn(1600, 2401)

torch.onnx.export(
    fnet,                                 
    dummy_input_fnet,                     
    "./onnx_models/fnet.onnx",                          
    input_names=['input'],                
    output_names=['output'],              
    opset_version=11                      
)

torch.onnx.export(
    updateformer,                         
    dummy_input_updateformer,             
    "./onnx_models/updateformer.onnx",                  
    input_names=['input'],                
    output_names=['output'],              
    opset_version=11                      
)

torch.onnx.export(
    corr_mlp,
    dummy_input_corr_mlp,
    "./onnx_models/corr_mlp.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)
print("ONNXファイルが正常に作成されました。")