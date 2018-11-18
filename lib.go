package pytorch

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g -Wno-sign-compare -Wno-unused-function -I/usr/local/cuda/include -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include
// #cgo LDFLAGS: -lstdc++ -L/usr/local/cuda/lib64  -lcuda -lcudart -lcublas -lcudnn -L/opt/libtorch/lib -ltorch -lcaffe2_gpu -lcaffe2 -lc10 -lcudart -lnvrtc-builtins -lnvrtc -lnvToolsExt -lgomp -lcuda
import "C"
