package pytorch

// #include "cbits/predictor.hpp"
// #include <stdlib.h>
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

func (p *Predictor) StartProfiling(name, metadata string) error {
	cname := C.CString(name)
	cmetadata := C.CString(metadata)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cmetadata))
	C.StartProfilingPytorch(p.ctx, cname, cmetadata)
	return nil

}

func (p *Predictor) EndProfiling() error {
	C.EndProfilingPytorch(p.ctx)
	return nil
}

func (p *Predictor) EnableProfiling() error {
	C.EnableProfilingPytorch(p.ctx)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.DisableProfilingPytorch(p.ctx)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.ReadProfilePytorch(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}
