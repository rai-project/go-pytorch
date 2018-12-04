package pytorch

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"context"
	"fmt" //"github.com/k0kubun/pp"
	"unsafe"

	"github.com/Unknwon/com"
	//"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
)

const (
	CPUMode = 0
	GPUMode = 1
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

	mode := CPUMode
	if options.UsesGPU() {
		if !nvidiasmi.HasGPU {
			return nil, errors.New("no GPU device")
		}
		SetUseGPU()
		mode = GPUMode
	} else {
		SetUseCPU()
	}

	return &Predictor{
		ctx: C.NewPytorch(
			C.CString(modelFile),
			C.int(options.BatchSize()),
			C.int(mode),
		),
		options: options,
	}, nil
}

func SetUseCPU() {
	C.SetModePytorch(C.int(CPUMode))
}

func SetUseGPU() {
	C.SetModePytorch(C.int(GPUMode))
}

func init() {
	C.InitPytorch()
}

func (p *Predictor) Predict(ctx context.Context, data []float32) error {

	if data == nil || len(data) < 1 {
		return fmt.Errorf("input nil or empty")
	}

	batchSize := p.options.BatchSize()
	width := C.GetWidthPytorch(p.ctx)
	height := C.GetHeightPytorch(p.ctx)
	channels := C.GetChannelsPytorch(p.ctx)
	shapeLen := int(width * height * channels)

	dataLen := len(data)

	inputCount := dataLen / shapeLen
	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))

	predictSpan, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer predictSpan.Finish()

	C.PredictPytorch(p.ctx, ptr)

	return nil
}

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]float32, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_predicted_output")
	defer span.Finish()

	batchSize := p.options.BatchSize()
	predLen := int(C.GetPredLenPytorch(p.ctx))
	length := batchSize * predLen

	cPredictions := C.GetPredictionsPytorch(p.ctx)
	if cPredictions == nil {
		return nil, errors.New("empty predictions")
	}

	slice := (*[1 << 30]float32)(unsafe.Pointer(cPredictions))[:length:length]
	//pp.Println(slice[:2])

	return slice, nil
}

func (p *Predictor) Close() {
	C.DeletePytorch(p.ctx)
}
