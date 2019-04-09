import pretrainedmodels as pm
import torch

# models giving an error
errored_model_name = ['fbresnet152', 'bninception', 'inceptionv4', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# collect all (model, pretrained) tuples
pm_args = []
for model_name in pm.model_names:
	for pretrained in pm.pretrained_settings[model_name]:
		if pretrained in ['imagenet', 'imagenet+5k']:
			pm_args.append([model_name, pretrained])

for i in range(len(pm_args)):
	# download model
	model_name = pm_args[i][0]
	pretrained_on = pm_args[i][1]
	model = pm.__dict__[model_name](num_classes=1000, pretrained=pretrained_on)
	model.eval()
	if model_name not in errored_model_name:
		# fetch input_size
		print("REFERENCE model - ", model_name)
		model_settings = pm.pretrained_settings[model_name]
		print(model_settings)
