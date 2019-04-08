package pytorch

// #include "cbits/predictor.hpp"
import "C"

import (
	"bytes"
	"reflect"
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

func getFlattenedLength(data []int64) int {
	res := 1
	for _, d := range data {
		res *= int(d)
	}
	return res
}

func tensorCtxToTensor(tensorCtx C.Torch_TensorContext) *tensor.Dense {
	var shapeLength C.ulong
	ptr := C.Torch_TensorValue(tensorCtx)
	cShape := C.Torch_TensorShape(tensorCtx, &shapeLength)
	ty := DType(C.Torch_TensorType(tensorCtx))

	runtime.KeepAlive(cShape)
	runtime.KeepAlive(ptr)

	cShapeSlice := (*[1 << 30]int64)(unsafe.Pointer(cShape))[:shapeLength:shapeLength]

	shape := tensor.Shape(toIntSlice(cShapeSlice))
	flattenedLength := getFlattenedLength(cShapeSlice)

	switch ty {
	case Byte:
		{
			cData := (*[1 << 30]uint8)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]uint8, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Byte,
				shape,
				tensor.WithBacking(data),
			)
		}
	case Char:
		{
			cData := (*[1 << 30]byte)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]byte, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Uint8,
				shape,
				tensor.WithBacking(data),
			)
		}
	case Int:
		{
			cData := (*[1 << 30]int)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]int, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Int,
				shape,
				tensor.WithBacking(data),
			)
		}
	case Long:
		{
			cData := (*[1 << 30]uint64)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]uint64, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Uint64,
				shape,
				tensor.WithBacking(data),
			)
		}
	case Float:
		{
			cData := (*[1 << 30]float32)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]float32, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Float32,
				shape,
				tensor.WithBacking(data),
			)
		}
	case Double:
		{
			cData := (*[1 << 30]float64)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]float64, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Float64,
				shape,
				tensor.WithBacking(data),
			)
		}
	default:
		panic("invalid data type")
	}
}

func fromType(ten *tensor.Dense) DType {
	for _, t := range types {
		if t.typ == ten.Dtype().Type {
			return DType(t.dataType)
		}
	}
	return UnknownType
}

func toTensorCtx(ten *tensor.Dense, device DeviceKind) C.Torch_TensorContext {
	shape := make([]int64, len(ten.Shape()))
	for ii, s := range ten.Shape() {
		shape[ii] = int64(s)
	}

	nbytes := ten.Dtype().Size() * uintptr(ten.DataSize())
	dataPtr := unsafe.Pointer(C.malloc(C.size_t(nbytes)))
	dataSlice := (*[1 << 30]byte)(dataPtr)[:nbytes:nbytes]

	buf := bytes.NewBuffer(dataSlice[:0:nbytes])
	encodeTensor(buf, reflect.ValueOf(ten.Data()), shape)

	return createTensor(dataPtr, shape, fromType(ten), device)
}

func ivalueToTensor(ctx C.Torch_IValue) []tensor.Tensor {
	if ctx.itype == C.Torch_IValueTypeTensor {
		tensorCtx := (C.Torch_TensorContext)(&ctx.data_ptr)
		tensr := tensorCtxToTensor(tensorCtx)
		return []tensor.Tensor{tensr}
	}

	if ctx.itype != C.Torch_IValueTypeTuple {
		panic("expecting a C.Torch_IValueTypeTuple type")
	}

	tuple := (*C.Torch_IValueTuple)(ctx.data_ptr)
	tupleLength := int(tuple.length)

	tupleSlice := (*[1 << 30]C.Torch_IValue)(unsafe.Pointer(tuple.values))[:tupleLength:tupleLength]

	res := []tensor.Tensor{}
	for _, elem := range tupleSlice {
		tensors := ivalueToTensor(elem)
		res = append(res, tensors...)
	}

	return res
}

func freeTuple(tuple *C.Torch_IValueTuple) {
	valuesSlice := (*[1 << 30]C.Torch_IValue)(unsafe.Pointer(tuple.values))[:tuple.length:tuple.length]
	freeIValues(valuesSlice)
	C.free(unsafe.Pointer(tuple.values))
	C.free(unsafe.Pointer(tuple))
}

func freeIValues(values []C.Torch_IValue) {
	for _, val := range values {
		if val.itype == C.Torch_IValueTypeTuple {
			freeTuple((*C.Torch_IValueTuple)(val.data_ptr))
		}
	}
}

func convertIValueToGoType(ival C.Torch_IValue) (interface{}, error) {
	if ival.itype == C.Torch_IValueTypeTensor {
		tensorContext := (C.Torch_TensorContext)(ival.data_ptr)
		return tensorWithContext(tensorContext), nil
	}

	if ival.itype == C.Torch_IValueTypeTuple {
		tuple := (*C.Torch_IValueTuple)(ival.data_ptr)
		return convertIValueTupleToTuple(tuple)
	}

	// TODO handle errors
	return nil, nil
}
