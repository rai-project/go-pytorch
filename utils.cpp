
#include "error.hpp"
#include "predictor.hpp"

extern void Torch_DeleteTensor(Torch_TensorContext ctx);
extern void Torch_TupleDelete(Torch_TupleContext tuple);

void Torch_IValueDelete(Torch_IValue val) {
  if (val.itype == Torch_IValueTypeTensor) {
    Torch_DeleteTensor((Torch_TensorContext)val.data_ptr);
    return;
  }
  if (val.itype == Torch_IValueTypeTuple) {
    Torch_TupleDelete((Torch_TupleContext)val.data_ptr);
    return;
  }
  return;
}