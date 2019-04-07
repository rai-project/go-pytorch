# Go Bindings for Pytorch

[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/rai-project.go-pytorch)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=10)

## Libtorch Installation

### Pre-built Binaries

Download the relevant `LibTorch` pre-built binary available on [Pytorch website](https://pytorch.org). Note that we provide the option of profiling through pytorch's in-built autograd profiler. Incidentally, Pytorch C++ frontend does not have access to the autograd profiler as per release `1.0.1`. Kindly download nightly build post March 24th 2019 to enable the profiling. Without profiling, our codebase should be compatible with prior versions.

### Build From Source

Kindly refer to `dockerfiles` to know how to build `LibTorch` from source. Note that one can also use `build_libtorch.py` script provided as part of the Pytorch repository to do the same.

## Build From Source using PIP

```
pip3 install torch torchvision
```

or

```
conda install pytorch-nightly -c pytorch
```

then build using

```
go build -tags=nogpu -tags=python
```

## Use Other Library Paths

We assume the default library location to be `/opt/libtorch`.
To use different library paths, change CGO_CFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS enviroment variables to add the corresponding framework library/non-library paths.

For example,

```
    export CGO_CFLAGS="${CGO_CFLAGS} -I /usr/local/cuda-9.2/include -I/usr/local/cuda-9.2/nvvm/include -I /usr/local/cuda-9.2/extras/CUPTI/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include/crt"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -I /usr/local/cuda-9.2/include -I/usr/local/cuda-9.2/nvvm/include -I /usr/local/cuda-9.2/extras/CUPTI/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include/crt"
    export CGO_LDFLAGS="${CGO_LDFLAGS} -L /usr/local/nvidia/lib64 -L /usr/local/cuda-9.2/nvvm/lib64 -L /usr/local/cuda-9.2/lib64 -L /usr/local/cuda-9.2/lib64/stubs -L /usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs/ -L /usr/local/cuda-9.2/lib64/stubs -L /usr/local/cuda-9.2/extras/CUPTI/lib64"
```

Run `go build` in to check the Libtorch installation and library paths set-up.

## Run Example

Make sure you have already [install mlmodelscope dependences](https://docs.mlmodelscope.org/installation/source/dependencies/) and [set up the external services](https://docs.mlmodelscope.org/installation/source/external_services/).

On linux, the default is to use GPU, if you don't have a GPU, do `go build -tags nogpu` instead of `go build`.

### batch

This example is to show how to use mlmodelscope tracing to profile the inference.

```
  cd example/batch
  go build
  ./batch
```

Then you can go to `localhost:16686` to look at the trace of that inference.

### batch_nvprof

You need GPU and CUDA to run this example. This example is to show how to use nvprof to profile the inference.

```
  cd example/batch_nvprof
  go build
  nvprof --profile-from-start off ./batch_nvprof
```

Refer to [Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html) on how to use nvprof.

## Credits

Parts of the implementation is borrowed from [orktes/go-torch](https://github.com/orktes/go-torch)
