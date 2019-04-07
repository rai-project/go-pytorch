#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorContext;

typedef enum { CPU_DEVICE_KIND = 0, CUDA_DEVICE_KIND = 1 } DeviceKind;

PredictorContext NewPytorch(char *model_file, int batch,
                          int mode);

void SetModePytorch(int mode);

void InitPytorch();

void AddFloat32PytorchPrediction(PredictorContext pred, int ii,
                                 float *inputData);
void AddFloat64PytorchPrediction(PredictorContext pred, int ii,
                                 double *inputData);

const int GetNumberofTensorsPytorch(PredictorContext pred);

const int *GetPredictionSizesPytorch(PredictorContext pred);

const float *GetPredictionsPytorch(PredictorContext pred);

void DeletePytorch(PredictorContext pred);

int GetWidthPytorch(PredictorContext pred);

int GetHeightPytorch(PredictorContext pred);

int GetChannelsPytorch(PredictorContext pred);

void SetDimensionsPytorch(PredictorContext pred, int channels, int height, int width, int batch);

int GetPredLenPytorch(PredictorContext pred);

void StartProfilingPytorch(PredictorContext pred, const char *name, const char *metadata);

void EndProfilingPytorch(PredictorContext pred);

void EnableProfilingPytorch(PredictorContext pred);

void DisableProfilingPytorch(PredictorContext pred);

char *ReadProfilePytorch(PredictorContext pred);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
