package pytorch

// #include "cbits/predictor.hpp"
// #include <stdlib.h>
//
import "C"
import (
	"unsafe"
)

// Error errors returned by torch functions
type Error struct {
	message string
}

func (te *Error) Error() string {
	return te.message
}

func checkError(err C.Torch_Error) *Error {
	if err.message != nil {
		defer C.free(unsafe.Pointer(err.message))
		return &Error{
			message: C.GoString(err.message),
		}
	}

	return nil
}

func HasError() bool {
	return int(C.Torch_HasError()) == 0
}

func GetErrorString() string {
	return C.GoString(C.Torch_GetErrorString())
}

func ResetError() {
	C.Torch_ResetError()
}

func PanicOnError() {
	msg := C.Torch_GetErrorString()
	if msg == nil {
		return
	}
	panic(C.GoString(msg))
}
