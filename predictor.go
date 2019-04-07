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
	ctx     C.PredictorContext
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
	C.Torch_PredictorSetMode(C.int(0))
}

func SetUseGPU() {
	C.Torch_PredictorSetMode(C.int(1))
}

func init() {
	C.InitPytorch()
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

	if cPredictions.itype == C.Torch_IValueTypeTensor {
		tensorCtx := (*C.Torch_TensorContext)(&cPredictions.data_ptr)
		tensor := createTensor(
			C.Torch_TensorValue(tensorCtx),
			C.Torch_TensorShape(tensorCtx),
			C.Torch_TensorType(predictionTensor),
		)
		return tensor, nil
	}

	// // TODO create variable number of slices = O(cNumOfSizes)
	// // creating <= 2 as of now
	// slice_sizes := (*[1 << 30]int32)(unsafe.Pointer(cSizes))[:cNumOfSizes]
	// slice_0 := (*[1 << 30]float32)(unsafe.Pointer(cPredictions))[:slice_sizes[0]]
	// pp.Println(slice_0[:2])
	// /*if cNumOfSizes >= 2 {
	// 	slice_1 := (*[1 << 30]float32)(unsafe.Pointer(cPredictions))[(slice_sizes)[:1]:(slice_sizes)[1:2]]
	// 	pp.Println(slice_1[:2])
	// }*/

	// //slice := (*[1 << 30]float32)(unsafe.Pointer(cPredictions))[:length:length]
	// //pp.Println(slice[:2])

	// // TODO returning first slices for now
	return nil, nil
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
