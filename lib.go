package pytorch

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O0 -Wall -g -Wno-sign-compare -Wno-unused-function  -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include
// #cgo LDFLAGS: -lstdc++ -L/opt/libtorch/lib -ltorch -lcaffe2 -lc10 -lgomp
// #cgo linux,amd64,!nogpu CXXFLAGS: -I/usr/local/cuda/include
// #cgo linux,amd64,!nogpu LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -L/opt/libtorch/lib -lcaffe2_gpu -lcudart -lnvrtc-builtins -lnvrtc -lnvToolsExt -lcuda
import "C"
