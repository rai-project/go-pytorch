package pytorch

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -g -O3
// #cgo CFLAGS: -I${SRCDIR}/cbits -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo LDFLAGS: -lstdc++ -ltorch -lc10 -lglog
// #cgo !python CXXFLAGS: -isystem /opt/libtorch/include
// #cgo !python CXXFLAGS: -isystem /opt/libtorch/include/torch/csrc/api/include
// #cgo !python CXXFLAGS: -isystem /opt/libtorch/include/torch/csrc
// #cgo !python LDFLAGS: -lgomp -L/opt/libtorch/lib
// #cgo linux,amd64,!nogpu CFLAGS: -I/usr/local/cuda/include
// #cgo linux,amd64,!nogpu LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcudnn -lcaffe2_nvrtc -lnvrtc-builtins -lnvrtc -lnvToolsExt -L/opt/libtorch/lib -lc10_cuda
// #cgo python CXXFLAGS: -isystem /usr/local/anaconda3/lib/python3.6/site-packages/torch/include/torch/csrc/api/include
// #cgo python CXXFLAGS: -isystem /usr/local/anaconda3/lib/python3.6/site-packages/torch/include
// #cgo python CXXFLAGS: -isystem /usr/local/anaconda3/lib/python3.6/site-packages/torch/include/torch/csrc
// #cgo darwin,python LDFLAGS: -L/usr/local/anaconda3/lib/python3.6/site-packages/torch/lib
import "C"
