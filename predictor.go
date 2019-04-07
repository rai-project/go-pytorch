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
	ctx          C.Torch_PredictorContext
	inputsLength int
	inputs       *C.Torch_TensorContext
	options      *options.Options
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

	device := CPUDeviceKind
	if options.UsesGPU() {
		if !nvidiasmi.HasGPU {
			return nil, errors.New("no GPU device")
		}
		device = CUDADeviceKind
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

func (p *Predictor) Predict(ctx context.Context, inputs []*tensor.Dense) error {
	defer PanicOnError()

	if len(inputs) < 1 {
		return fmt.Errorf("input nil or empty")
	}

	inputsLength := len(inputs)
	cInputs := (*C.Torch_TensorContext)(C.malloc(C.size_of_tensor_ctx * C.ulong(inputsLength)))

	p.inputs = cInputs
	p.inputsLength = inputsLength

	inputSlice := (*[1 << 30]C.Torch_TensorContext)(unsafe.Pointer(cInputs))[:inputsLength:inputsLength]

	for ii, input := range inputs {
		inputSlice[ii] = toTensorCtx(input)
	}

	predictSpan, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer predictSpan.Finish()

	C.Torch_PredictorRun(p.ctx, cInputs, C.int(inputsLength))

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

func (p *Predictor) freeInputs() {
	if p.inputs != nil {
		return
	}
	inputSlice := (*[1 << 30]C.Torch_TensorContext)(unsafe.Pointer(p.inputs))[:p.inputsLength:p.inputsLength]
	for _, input := range inputSlice {
		C.Torch_DeleteTensor(input)
	}

	C.free(unsafe.Pointer(p.inputs))
}

func (p *Predictor) finalize() {
	p.freeInputs()
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
