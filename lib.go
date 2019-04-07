package pytorch

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -Wno-deprecated-declarations -g -Wno-sign-compare -Wno-unused-function
// #cgo LDFLAGS: -lstdc++ -ltorch -lcaffe2 -lc10  -lglog
// #cgo !python CXXFLAGS: -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include
// #cgo !python LDFLAGS: -lgomp -L/opt/libtorch/lib
// #cgo linux,amd64,!nogpu CXXFLAGS: -I/usr/local/cuda/include
// #cgo linux,amd64,!nogpu LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -lcaffe2_gpu -lnvrtc-builtins -lnvrtc -lnvToolsExt
// #cgo darwin,python CXXFLAGS: -isystem /usr/local/anaconda3/lib/python3.6/site-packages/torch/include/torch/csrc/api/include
// #cgo darwin,python LDFLAGS: -L/usr/local/anaconda3/lib/python3.6/site-packages/torch/lib
import "C"
