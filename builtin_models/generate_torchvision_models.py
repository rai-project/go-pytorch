import torch
import torchvision
import types
import sys


models = {model: getattr(torchvision.models, model) for model in dir(torchvision.models)
if isinstance(getattr(torchvision.models, model), types.FunctionType)}

fake_input = torch.rand(1, 3, 224, 224)
for name, f in models.items():
	try:
		print("Processing " + name)
		model = f(pretrained=True)
		traced_script_module = torch.jit.trace(model, fake_input, check_trace=False)
		traced_script_module.save("/data/models/" + name + ".pt")
	except:
		print("Unexpected error:", sys.exc_info())
		pass
