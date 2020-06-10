#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#define _GLIBCXX_USE_CXX11_ABI 1

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus

#ifndef PROFILING_ENABLED
#ifndef PROFILING_DISABLED
// profiling is enabled only if you have the include file
#if __has_include(<autograd/profiler.h>)
#include <autograd/profiler.h>
#include <jit/code_template.h>
#define PROFILING_ENABLED 1
#endif  //  __has_include(<../../autograd/profiler.h>)
#endif  // PROFILING_DISABLED
#endif  // PROFILING_ENABLED

#include <torch/script.h>
#include <torch/torch.h>

struct Torch_Tensor {
  torch::Tensor tensor;
};

extern "C" {

#endif  // __cplusplus

typedef enum Torch_DataType {
  Torch_Unknown = 0,
  Torch_Byte = 1,
  Torch_Char = 2,
  Torch_Short = 3,
  Torch_Int = 4,
  Torch_Long = 5,
  Torch_Half = 6,
  Torch_Float = 7,
  Torch_Double = 8,

} Torch_DataType;

typedef enum Torch_IValueType {
  Torch_IValueTypeUnknown = 0,
  Torch_IValueTypeTensor = 1,
  Torch_IValueTypeTuple = 2,
} Torch_IValueType;

typedef struct Torch_IValue {
  Torch_IValueType itype;
  void* data_ptr;
} Torch_IValue;

typedef struct Torch_IValueTuple {
  Torch_IValue* values;
  size_t length;
} Torch_IValueTuple;

typedef struct Torch_ModuleMethodArgument {
  char* name;
  char* typ;
  // Torch_TensorContext default_value;
  // Torch_DataType type;
} Torch_ModuleMethodArgument;

typedef struct Torch_Error {
  char* message;
} Torch_Error;

extern Torch_Error Torch_GlobalError;

typedef enum { UNKNOWN_DEVICE_KIND = -1, CPU_DEVICE_KIND = 0, CUDA_DEVICE_KIND = 1 } Torch_DeviceKind;

typedef void* Torch_PredictorContext;
typedef Torch_IValueTuple* Torch_TupleContext;
typedef void* Torch_TensorContext;
typedef void* Torch_JITModuleContext;
typedef void* Torch_JITModuleMethodContext;

void InitPytorch();

// Predictor

Torch_PredictorContext Torch_NewPredictor(const char* model_file, Torch_DeviceKind mode);

void Torch_PredictorAddInput(Torch_PredictorContext pred, Torch_DataType ty, void* data);

void Torch_PredictorRun(Torch_PredictorContext pred, Torch_TensorContext* cInputs, int inputLength);

int Torch_PredictorNumOutputs(Torch_PredictorContext pred);

Torch_IValue Torch_PredictorGetOutput(Torch_PredictorContext pred);

void Torch_PredictorDelete(Torch_PredictorContext pred);

// Error

char Torch_HasError();

const char* Torch_GetErrorString();

void Torch_ResetError();

// IValue

void Torch_IValueDelete(Torch_IValue val);

// Tuple

int Torch_TupleLength(Torch_TupleContext tup);
Torch_IValue Torch_TupleElement(Torch_TupleContext tup, int elem);
void Torch_TupleDelete(Torch_TupleContext tup);

// Tensor
Torch_TensorContext Torch_NewTensor(void* data, int64_t* dimensions, int n_dim, Torch_DataType dtype,
                                    Torch_DeviceKind device);
void* Torch_TensorValue(Torch_TensorContext ctx);
Torch_DataType Torch_TensorType(Torch_TensorContext ctx);
int64_t* Torch_TensorShape(Torch_TensorContext ctx, size_t* dims);
void Torch_DeleteTensor(Torch_TensorContext ctx);

void Torch_PrintTensors(Torch_TensorContext* tensors, size_t input_size);

// Profile
void Torch_ProfilingStart(Torch_PredictorContext pred, const char* name, const char* metadata);

void Torch_ProfilingEnd(Torch_PredictorContext pred);

void Torch_ProfilingEnable(Torch_PredictorContext pred);

void Torch_ProfilingDisable(Torch_PredictorContext pred);

char* Torch_ProfilingRead(Torch_PredictorContext pred);

int64_t Torch_ProfilingGetStartTime(Torch_PredictorContext pred);

// JIT
#ifdef ENABLE_PYTROCH_JIT
Torch_JITModuleContext Torch_CompileTorchScript(char* script, Torch_Error* error);
Torch_JITModuleContext Torch_LoadJITModule(char* path, Torch_Error* error);
void Torch_ExportJITModule(Torch_JITModuleContext ctx, char* path, Torch_Error* error);
Torch_JITModuleMethodContext Torch_JITModuleGetMethod(Torch_JITModuleContext ctx, char* method, Torch_Error* error);
char** Torch_JITModuleGetMethodNames(Torch_JITModuleContext ctx, size_t* len);
Torch_IValue Torch_JITModuleMethodRun(Torch_JITModuleMethodContext ctx, Torch_IValue* inputs, size_t input_size,
                                      Torch_Error* error);
Torch_ModuleMethodArgument* Torch_JITModuleMethodArguments(Torch_JITModuleMethodContext ctx, size_t* res_size);
Torch_ModuleMethodArgument* Torch_JITModuleMethodReturns(Torch_JITModuleMethodContext ctx, size_t* res_size);
void Torch_DeleteJITModuleMethod(Torch_JITModuleMethodContext ctx);
void Torch_DeleteJITModule(Torch_JITModuleContext ctx);
#endif  // ENABLE_PYTROCH_JIT
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
