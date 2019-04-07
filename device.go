package pytorch

// #include "cbits/predictor.hpp"
import "C"

type DeviceKind C.Torch_DeviceKind

const (
	CPUDeviceKind  DeviceKind = C.CPU_DEVICE_KIND
	CUDADeviceKind DeviceKind = C.CUDA_DEVICE_KIND
)
