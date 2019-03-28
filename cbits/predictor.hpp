#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

typedef void *PredictorContext;

PredictorContext NewPytorch(char *model_file, int batch, int mode);

void SetModePytorch(int mode);

void InitPytorch();

void AddFloat32PytorchPrediction(PredictorContext pred, int ii,
                                 float *inputData);
void AddFloat64PytorchPrediction(PredictorContext pred, int ii,
                                 double *inputData);

const float *GetFloat32PredictionsPytorch(PredictorContext pred, int ii);
const double *GetFloat64PredictionsPytorch(PredictorContext pred, int ii);

void DeletePytorch(PredictorContext pred);

int GetWidthPytorch(PredictorContext pred);

int GetHeightPytorch(PredictorContext pred);

int GetChannelsPytorch(PredictorContext pred);

int GetPredLenPytorch(PredictorContext pred);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
