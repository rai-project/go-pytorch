package pytorch

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
//
// size_t size_of_tensor_ctx = sizeof(Torch_TensorContext);
import "C"
import (
	"context"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	"gorgonia.org/tensor"
)

type Predictor struct {
	ctx     C.Torch_PredictorContext
	inputs  []C.Torch_TensorContext
	options *options.Options
}

func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
	defer PanicOnError()

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_new")
	defer span.Finish()

	options := options.New(opts...)
	modelFile := string(options.Graph())
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}

	device := fromDevice(options)
	if device == UnknownDeviceKind {
		return nil, errors.New("invalid device")
	}

	cModelFile := C.CString(modelFile)
	defer C.free(unsafe.Pointer(cModelFile))

	pred := &Predictor{
		ctx: C.Torch_NewPredictor(
			cModelFile,
			C.Torch_DeviceKind(device),
		),
		options: options,
	}

	runtime.SetFinalizer(pred, (*Predictor).finalize)

	return pred, nil
}

func fromDevice(opts *options.Options) DeviceKind {
	device := CPUDeviceKind
	if opts.UsesGPU() {
		if !nvidiasmi.HasGPU {
			return UnknownDeviceKind
		}
		device = CUDADeviceKind
	}
	return device
}

func (p *Predictor) Predict(ctx context.Context, inputs []tensor.Tensor) error {
	defer PanicOnError()

	if len(inputs) < 1 {
		return fmt.Errorf("input nil or empty")
	}

	inputsLength := len(inputs)
	inputSlice := make([]C.Torch_TensorContext, inputsLength)

	for ii, input := range inputs {
		dense, ok := input.(*tensor.Dense)
		if !ok {
			return errors.New("expecting a dense tensor")
		}
		inputSlice[ii] = toTensorCtx(dense, fromDevice(p.options))
	}

	predictSpan, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer predictSpan.Finish()

	C.Torch_PredictorRun(p.ctx, &inputSlice[0], C.int(inputsLength))

	return nil
}

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]tensor.Tensor, error) {
	defer PanicOnError()

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_predicted_output")
	defer span.Finish()

	cNumOutputs := int(C.Torch_PredictorNumOutputs(p.ctx))
	if cNumOutputs == 0 {
		return nil, errors.New("zero number of tensors")
	}

	cPredictions := C.Torch_PredictorGetOutput(p.ctx)
	defer C.Torch_IValueDelete(cPredictions)

	if cPredictions.itype == C.Torch_IValueTypeUnknown {
		return nil, errors.New("empty predictions")
	}

	return ivalueToTensor(cPredictions), nil
}

func (p *Predictor) finalize() {
	for _, input := range p.inputs {
		C.Torch_DeleteTensor(input)
	}
	if p.ctx != nil {
		C.Torch_PredictorDelete(p.ctx)
	}
	p.ctx = nil
}

func (p *Predictor) Close() {
	p.finalize()
}

func init() {
	C.InitPytorch()
}
