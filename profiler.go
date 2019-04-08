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
	C.Torch_ProfilingStart(p.ctx, cname, cmetadata)
	return nil

}

func (p *Predictor) EndProfiling() error {
	C.Torch_ProfilingEnd(p.ctx)
	return nil
}

func (p *Predictor) EnableProfiling() error {
	C.Torch_ProfilingEnable(p.ctx)
	return nil
}

func (p *Predictor) DisableProfiling() error {
	C.Torch_ProfilingDisable(p.ctx)
	return nil
}

func (p *Predictor) ReadProfile() (string, error) {
	cstr := C.Torch_ProfilingRead(p.ctx)
	if cstr == nil {
		return "", errors.New("failed to read nil profile")
	}
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr), nil
}
