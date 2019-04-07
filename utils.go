package pytorch

// #include "cbits/predictor.hpp"
import "C"

import (
	"runtime"
	"unsafe"

	"gorgonia.org/tensor"
)

func toIntSlice(data []int64) []int {
	res := make([]int, len(data))
	for ii, d := range data {
		res[ii] = int(d)
	}
	return res
}

func getFlattenedLength(data []int) int {
	res := 1
	for _, d := range data {
		res *= data
	}
	return res
}

func tensorCtxToTensor(ctx C.Torch_TensorContext) tensor.Tensor {
	var shapeLength C.ulong
	ptr := C.Torch_TensorValue(tensorCtx)
	cShape := C.Torch_TensorShape(ctx, &shapeLength)
	ty := C.Torch_TensorType(predictionTensor)

	runtime.KeepAlive(shape)
	runtime.KeepAlive(ptr)

	cShapeSlice := (*[1 << 30]int64)(unsafe.Pointer(cShape))[:shapeLength:shapeLength]

	shape := toIntSlice(shape)
	flattenedLength := getFlattenedLength(shapeSlice)

	switch ty {
	case Byte:
		{
			data := (*[1 << 30]uint8)(unsafe.Pointer(cShape))[:flattenedLength:flattenedLength]
			return tensor.NewDense(
				tensor.WithShape(shape...),
				tensor.WithBacking(data),
				tensor.Of(tensor.Uint8),
			)
		}
	case Char:
		{
			data := (*[1 << 30]byte)(unsafe.Pointer(cShape))[:flattenedLength:flattenedLength]
			return tensor.NewDense(
				tensor.WithShape(shape...),
				tensor.WithBacking(data),
				tensor.Of(tensor.Uint8),
			)
		}
	case Short:
		{
			data := (*[1 << 30]uint16)(unsafe.Pointer(cShape))[:flattenedLength:flattenedLength]
			return tensor.NewDense(
				tensor.WithShape(shape...),
				tensor.WithBacking(data),
				tensor.Of(tensor.Uint16),
			)
		}
	case Int:
		{
			data := (*[1 << 30]int)(unsafe.Pointer(cShape))[:flattenedLength:flattenedLength]
			return tensor.NewDense(
				tensor.WithShape(shape...),
				tensor.WithBacking(data),
				tensor.Of(tensor.Int),
			)
		}
	case Long:
		{
			data := (*[1 << 30]uint64)(unsafe.Pointer(cShape))[:flattenedLength:flattenedLength]
			return tensor.NewDense(
				tensor.WithShape(shape...),
				tensor.WithBacking(data),
				tensor.Of(tensor.Uint64),
			)
		}
	case Float:
		{
			data := (*[1 << 30]float32)(unsafe.Pointer(cShape))[:flattenedLength:flattenedLength]
			return tensor.NewDense(
				tensor.WithShape(shape...),
				tensor.WithBacking(data),
				tensor.Of(tensor.Float32),
			)
		}
	case Double:
		{
			data := (*[1 << 30]float64)(unsafe.Pointer(cShape))[:flattenedLength:flattenedLength]
			return tensor.NewDense(
				tensor.WithShape(shape...),
				tensor.WithBacking(data),
				tensor.Of(tensor.Float64),
			)
		}
	default:
		panic("invalid data type")
	}
}

func ivalueToTensor(ctx C.Torch_IValue) []tensor.Tensor {
	if ctx.itype == C.Torch_IValueTypeTensor {
		tensorCtx := (*C.Torch_TensorContext)(&cPredictions.data_ptr)
		tensr := tensorCtxToTensor(tensorCtx)
		return []tensor.Tensor{tensr}
	}

	if ctx.itype != C.Torch_IValueTypeTuple {
		panic("expecting a C.Torch_IValueTypeTuple type")
	}

	tupleCtx := (*C.Torch_TupleContext)(&cPredictions.data_ptr)
	tupleLength := int(C.Torch_TupleLength(tupleCtx))

	res := []tensor.Tensor{}
	for ii := 0; ii < tupleLength; ii++ {
		ival := ivalueToTensor(C.Torch_TupleElement(tupleCtx, ii))
		res[ii] = append(res, ival...)
	}

	return res, nil
}
