package pytorch

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"context"
	"fmt"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
)

type Device int

const (
	CPUDevice  Device = Device(C.CPU_DEVICE_KIND)
	CUDADevice        = Device(C.CUDA_DEVICE_KIND)
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

	device := CPUDevice
	if options.UsesGPU() {
		if !nvidiasmi.HasGPU {
			return nil, errors.New("no GPU device")
		}
		device = CUDADevice
	}

	return &Predictor{
		ctx: C.Torch_NewPredictor(
			C.CString(modelFile),
			C.int(options.BatchSize()),
			C.int(device),
		),
		options: options,
	}, nil
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

	// dims = [len(gotensors), Shape[0] == height, Shape[1] == width, Shape[2] == channels]
	dataLen := dims[0]
	height := dims[1]
	width := dims[2]
	channels := dims[3]
	C.SetDimensionsPytorch(p.ctx, C.int(channels), C.int(height), C.int(width), C.int(batchSize))

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

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]float32, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_predicted_output")
	defer span.Finish()

	//batchSize := p.options.BatchSize()
	//predLen := int(C.GetPredLenPytorch(p.ctx))
	//length := batchSize * predLen
	cSizes := C.Torch_PredictorOutput(p.ctx)
	if cSizes == nil {
		return nil, errors.New("empty sizes")
	}

	cNumOfSizes := C.Torch_PredictorNumOutputs(p.ctx)
	if cNumOfSizes == 0 {
		return nil, errors.New("zero number of tensors")
	}

	cPredictions := C.GetPredictionsPytorch(p.ctx)
	if cPredictions == nil {
		return nil, errors.New("empty predictions")
	}

	// TODO create variable number of slices = O(cNumOfSizes)
	// creating <= 2 as of now
	slice_sizes := (*[1 << 30]int32)(unsafe.Pointer(cSizes))[:cNumOfSizes]
	slice_0 := (*[1 << 30]float32)(unsafe.Pointer(cPredictions))[:slice_sizes[0]]
	pp.Println(slice_0[:2])
	/*if cNumOfSizes >= 2 {
		slice_1 := (*[1 << 30]float32)(unsafe.Pointer(cPredictions))[(slice_sizes)[:1]:(slice_sizes)[1:2]]
		pp.Println(slice_1[:2])
	}*/

	//slice := (*[1 << 30]float32)(unsafe.Pointer(cPredictions))[:length:length]
	//pp.Println(slice[:2])

	// TODO returning first slices for now
	return slice_0, nil
}

func (p *Predictor) Close() {
	C.Torch_PredictorDelete(p.ctx)
}
