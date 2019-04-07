
#include "error.hpp"
#include "predictor.hpp"

extern void Torch_IValueDelete(Torch_IValue val);

int Torch_TupleLength(Torch_TupleContext tuple) {
  if (tuple == nullptr) {
    return 0;
  }
  return tuple->length;
}

Torch_IValue* Torch_TupleData(Torch_TupleContext tuple) { return reinterpret_cast<Torch_IValue*>(tuple->values); }

Torch_IValue Torch_TupleElement(Torch_TupleContext tuple, int elem) {
  if (elem >= Torch_TupleLength(tuple)) {
    return Torch_IValue{};
  }
  Torch_IValue* data = Torch_TupleData(tuple);

  return data[elem];
}

void Torch_TupleDelete(Torch_TupleContext tuple) {
  if (tuple == nullptr) {
    return;
  }
  const auto len = Torch_TupleLength(tuple);
  Torch_IValue* data = Torch_TupleData(tuple);
  for (int ii = 0; ii < len; ii++) {
    Torch_IValueDelete(data[ii]);
  }
  free(tuple);
}