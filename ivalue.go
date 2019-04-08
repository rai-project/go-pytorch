package pytorch

// #include "cbits/predictor.hpp"
// #include <stdlib.h>
// #include <stdint.h>
//
//
import "C"
import (
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
