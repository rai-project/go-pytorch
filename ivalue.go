package pytorch

// #include "cbits/predictor.hpp"
// #include <stdlib.h>
// #include <stdint.h>
//
// size_t size_of_ivalue_tuple = sizeof(Torch_IValueTuple);
// size_t size_of_ivalue = sizeof(Torch_IValue);
//
import "C"
import (
	"fmt"
	"unsafe"
)

func convertIValueTupleToTuple(tuple *C.Torch_IValueTuple) (Tuple, error) {
	valuesSlice := (*[1 << 30]C.Torch_IValue)(unsafe.Pointer(tuple.values))[:tuple.length:tuple.length]

	goTuple := make(Tuple, len(valuesSlice))

	for i, ival := range valuesSlice {
		var err error
		goTuple[i], err = convertIValueToGoType(ival)
		if err != nil {
			return nil, err
		}
	}

	return goTuple, nil
}

func convertGoValueToIValue(val interface{}) (C.Torch_IValue, error) {
	switch v := val.(type) {
	case *Tensor:
		return C.Torch_IValue{
			itype:    C.Torch_IValueTypeTensor,
			data_ptr: unsafe.Pointer(v.context),
		}, nil
	case Tuple:
		tuple := (*C.Torch_IValueTuple)(C.malloc((C.size_t)(C.size_of_ivalue_tuple)))
		tuple.values = (*C.Torch_IValue)(C.malloc(C.size_of_ivalue * C.ulong(len(v))))
		tuple.length = C.ulong(len(v))

		valuesSlice := (*[1 << 30]C.Torch_IValue)(unsafe.Pointer(tuple.values))[:tuple.length:tuple.length]

		for i, val := range v {
			var err error
			valuesSlice[i], err = convertGoValueToIValue(val)
			if err != nil {
				return C.Torch_IValue{}, err
			}
		}

		return C.Torch_IValue{
			itype:    C.Torch_IValueTypeTuple,
			data_ptr: unsafe.Pointer(tuple),
		}, nil
	default:
		return C.Torch_IValue{}, fmt.Errorf("invalid input type for run %T", val)
	}
}
