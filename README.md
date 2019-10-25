# go-pytorch

[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/rai-project.go-pytorch)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=10)
[![Build Status](https://travis-ci.org/rai-project/go-pytorch.svg?branch=master)](https://travis-ci.org/rai-project/go-pytorch)
[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/go-mxnet)](https://goreportcard.com/report/github.com/rai-project/go-pytorch)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/go-pytorch:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/go-pytorch:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-pytorch:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/go-pytorch:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-pytorch:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/go-pytorch:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-pytorch:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/go-pytorch:amd64-gpu-latest 'Get your own version badge on microbadger.com')

Go binding for Pytorch C++ API.
This is used by the [Pytorch agent](https://github.com/rai-project/pytorch) in [MLModelScope](mlmodelscope.org) to perform model inference in Go.

## Installation

Download and install go-pytorch:

```
go get -v github.com/rai-project/go-pytorch
```

The binding requires Pytorch C++ (libtorch) and other Go packages.

### Pytorch C++ (libtorch) Library

The Pytorch C++ library is expected to be under `/opt/libtorch`.

To install Pytorch C++ on your system, you can

1. download pre-built binary from [Pytorch website](https://pytorch.org): Choose `Pytorch Build = Stable (1.3)`, `Your OS = <fill>`, `Package = LibTorch`, `Language = C++` and `CUDA = <fill>`. Then download `cxx11 ABI` version. Unzip the packaged directory and copy to `/opt/libtorch` (or modify the corresponding `CFLAGS` and `LDFLAGS` paths if using a custom location).

2. build it from source: Refer to our [scripts](scripts) or the `LIBRARY INSTALLATION` section in the [dockefiles](dockerfiles).

- The default blas is OpenBLAS.
  The default OpenBLAS path for macOS is `/usr/local/opt/openblas` if installed throught homebrew (openblas is keg-only, which means it was not symlinked into /usr/local, because macOS provides BLAS and LAPACK in the Accelerate framework).

- The default pytorch C++ installation path is `/opt/libtorch` for linux, darwin and ppc64le without powerai

- The default CUDA path is `/usr/local/cuda`

See [lib.go](lib.go) for details.

If you get an error about not being able to write to `/opt` then perform the following

```
sudo mkdir -p /opt/libtorch
sudo chown -R `whoami` /opt/libtorch
```

If you are using Pytorch docker images or other libary paths, change CGO_CFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS enviroment variables. Refer to [Using cgo with the go command](https://golang.org/cmd/cgo/#hdr-Using_cgo_with_the_go_command).

For example,

```
    export CGO_CFLAGS="${CGO_CFLAGS} -I/tmp/libtorch/include"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -I/tmp/libtorch/include"
    export CGO_LDFLAGS="${CGO_LDFLAGS} -L/tmp/libtorch/lib"
```
### Go Packages

You can install the dependency through `go get`.

```
cd $GOPATH/src/github.com/rai-project/go-pytorch
go get -u -v ./...
```

Or use [Dep](https://github.com/golang/dep).

```
dep ensure -v
```

This installs the dependency in `vendor/`.

### Configure Environmental Variables

Configure the linker environmental variables since the Pytorch C++ library is under a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file

Linux
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/libtorch/lib
export LD_LIBRARY_PATH=/opt/libtorch/lib:$DYLD_LIBRARY_PATH

```

macOS
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/libtorch/lib
export DYLD_LIBRARY_PATH=/opt/libtorch/lib:$DYLD_LIBRARY_PATH
```
## Check the Build

Run `go build` in to check the dependences installation and library paths set-up.
On linux, the default is to use GPU, if you don't have a GPU, do `go build -tags nogpu` instead of `go build`.

**_Note_** : The CGO interface passes go pointers to the C API. This is an error by the CGO runtime. Disable the error by placing

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`

## Examples

Examples of using the Go Pytorch binding to do model inference are under [examples](examples)

### batch_mlmodelscope

This example shows how to use the MLModelScope tracer to profile the inference.

Refer to [Set up the external services](https://docs.mlmodelscope.org/installation/source/external_services/) to start the tracer.

Then run the example by

```
  cd example/batch_mlmodelscope
  go build
  ./batch
```

Now you can go to `localhost:16686` to look at the trace of that inference.

### batch_nvprof

This example shows how to use nvprof to profile the inference. You need GPU and CUDA to run this example.

```
  cd example/batch_nvprof
  go build
  nvprof --profile-from-start off ./batch_nvprof
```

Refer to [Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html) for using nvprof.

## Credits

Parts of the implementation have been borrowed from [orktes/go-torch](https://github.com/orktes/go-torch)
