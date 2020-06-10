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
	"strings"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	cupti "github.com/rai-project/go-cupti"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	"gorgonia.org/tensor"
)

type Predictor struct {
	ctx     C.Torch_PredictorContext
	inputs  []C.Torch_TensorContext
	options *options.Options
	cu      *cupti.CUPTI
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

	return pred, GetError()
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

	predictSpan, ctx := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer predictSpan.Finish()

	if p.options.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		p.EnableProfiling()
		err := p.StartProfiling("pytorch", "predict")
		if err != nil {
			log.WithError(err).WithField("framework", "pytorch").Error("unable to start framework profiling")
		} else {
			defer func() {
				p.EndProfiling()

				start_time := int64(C.Torch_ProfilingGetStartTime(p.ctx))

				profBuffer, err := p.ReadProfile()
				if err != nil {
					pp.Println(err)
					return
				}
				t, err := NewTrace(profBuffer, start_time)
				if err != nil {
					panic(err)
					return
				}
				t.Publish(ctx, tracer.FRAMEWORK_TRACE)
				p.DisableProfiling()
			}()
		}
	}

	err := p.cuptiStart(ctx)
	if err != nil {
		return err
	}

	C.Torch_PredictorRun(p.ctx, &inputSlice[0], C.int(inputsLength))

	p.cuptiClose()

	return GetError()
}

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]tensor.Tensor, error) {
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

	res := ivalueToTensor(cPredictions)
	if err := GetError(); err != nil {
		return nil, err
	}

	return res, nil
}

func (p *Predictor) finalize() {
	if p == nil {
		return
	}
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

func (p *Predictor) cuptiStart(ctx context.Context) error {
	if p.options.TraceLevel() < tracer.SYSTEM_LIBRARY_TRACE {
		return nil
	}
	metrics := []string{}
	if p.options.GPUMetrics() != "" {
		metrics = strings.Split(p.options.GPUMetrics(), ",")
	}

	cu, err := cupti.New(cupti.Context(ctx),
		cupti.SamplingPeriod(0),
		cupti.Metrics(metrics),
	)
	if err != nil {
		return err
	}

	p.cu = cu
	return nil
}

func (p *Predictor) cuptiClose() {
	if p.cu == nil {
		return
	}
	p.cu.Wait()
	p.cu.Close()
	p.cu = nil
}

func init() {
	C.InitPytorch()
}
