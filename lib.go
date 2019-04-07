package pytorch

// #cgo CXXFLAGS: -std=c++14 -I${SRCDIR}/cbits
// #cgo CFLAGS: -I${SRCDIR}/cbits -O0 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo LDFLAGS: -lstdc++ -ltorch -lcaffe2 -lc10  -lglog
// #cgo !python CXXFLAGS: -I/opt/libtorch/include -I/opt/libtorch/include/torch/csrc/api/include
// #cgo !python LDFLAGS: -lgomp -L/opt/libtorch/lib
// #cgo linux,amd64,!nogpu CFLAGS: -I/usr/local/cuda/include
// #cgo linux,amd64,!nogpu LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -lcaffe2_gpu -lnvrtc-builtins -lnvrtc -lnvToolsExt
// #cgo python CXXFLAGS: -isystem /usr/local/anaconda3/lib/python3.6/site-packages/torch/include/torch/csrc/api/include
// #cgo python CXXFLAGS: -isystem /usr/local/anaconda3/lib/python3.6/site-packages/torch/include
// #cgo darwin,python LDFLAGS: -L/usr/local/anaconda3/lib/python3.6/site-packages/torch/lib
import "C"
