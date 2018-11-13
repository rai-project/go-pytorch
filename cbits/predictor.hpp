#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorContext;

PredictorContext NewPytorch(char *model_file, int batch,
                          int mode);

void SetModePytorch(int mode);

void InitPytorch();

void PredictPytorch(PredictorContext pred, float *imageData);

const float *GetPredictionsPytorch(PredictorContext pred);

void DeletePytorch(PredictorContext pred);

void StartProfilingPytorch(PredictorContext pred, const char *name,
                         const char *metadata);

void EndProfilingPytorch(PredictorContext pred);

void DisableProfilingPytorch(PredictorContext pred);

char *ReadProfilePytorch(PredictorContext pred);

int GetWidthPytorch(PredictorContext pred);

int GetHeightPytorch(PredictorContext pred);

int GetChannelsPytorch(PredictorContext pred);

int GetPredLenPytorch(PredictorContext pred);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
