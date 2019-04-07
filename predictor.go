package pytorch

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
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
	options *options.Options
}

func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
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

	pred := &Predictor{
		ctx: C.Torch_NewPredictor(
			C.CString(modelFile),
			C.int(options.BatchSize()),
			device,
		),
		options: options,
	}

	runtime.SetFinalizer(pred, (*Predictor).finalize)

	return pred, nilf
}

func SetUseCPU() {
	C.Torch_PredictorSetMode(CPUDeviceKind)
}

func SetUseGPU() {
	C.Torch_PredictorSetMode(CUDADeviceKind)
}

func (p *Predictor) Predict(ctx context.Context, data []float32, dims []int) error {

	if data == nil || len(data) < 1 {
		return fmt.Errorf("input nil or empty")
	}

	batchSize := p.options.BatchSize()

	shapeLen := int(channels * width * height)
	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))

	predictSpan, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer predictSpan.Finish()

	C.Torch_PredictorRun(p.ctx, ptr)

	return nil
}

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]tensor.Tensor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_predicted_output")
	defer span.Finish()

	cNumOutputs := int(C.Torch_PredictorNumOutputs(p.ctx))
	if cNumOfSizes == 0 {
		return nil, errors.New("zero number of tensors")
	}

	cPredictions := C.Torch_PredictorGetOutput(p.ctx)
	if cPredictions == nil {
		return nil, errors.New("empty predictions")
	}

	defer C.Torch_IValueDelete(cPredictions)

	if cPredictions.itype == C.Torch_IValueTypeTensor {
		tensorCtx := (*C.Torch_TensorContext)(&cPredictions.data_ptr)
		tensr := tensorCtxToTensor(tensorCtx)
		return []tensor.Tensor{tensr}, nil
	}

	if cPredictions.itype != C.Torch_IValueTypeTuple {
		return nil, errors.New("expecting a C.Torch_IValueTypeTuple type")
	}

	tupleCtx := (*C.Torch_TupleContext)(&cPredictions.data_ptr)
	tupleLength := int(C.Torch_TupleLength(tupleCtx))

	res := make([]tensor.Tensor, tupleLength)
	for ii := 0; ii < tupleLength; ii++ {
		tensr := tensorCtxToTensor(C.Torch_TupleElement(tupleCtx, ii))
		res[ii] = tensr
	}

	return res, nil
}

func (p *Predictor) finalize() {
	if p.ctx == nil {
		return
	}
	C.Torch_PredictorDelete(p.ctx)
	p.ctx = nil
}

func (p *Predictor) Close() {
	p.finalize()
}

func init() {
	C.InitPytorch()
}
