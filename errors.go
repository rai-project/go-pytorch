package pytorch

// #include "cbits/predictor.hpp"
// #include <stdlib.h>
//
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
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
	return int(C.Torch_HasError()) == 1
}

func GetErrorString() string {
	msg := C.Torch_GetErrorString()
	if msg == nil {
		return ""
	}
	return C.GoString(msg)
}

func GetError() error {
	if !HasError() {
		return nil
	}
	err := errors.New(GetErrorString())
	ResetError()
	return err
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
