

#include "error.hpp"
#include "predictor.hpp"

#include <algorithm>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>


torch::TensorOptions Torch_ConvertDataTypeToOptions(Torch_DataType dtype) {
  torch::TensorOptions options;
  switch (dtype) {
    case Torch_Byte:
      options = torch::TensorOptions(torch::kByte);
      break;
    case Torch_Char:
      options = torch::TensorOptions(torch::kChar);
      break;
    case Torch_Short:
      options = torch::TensorOptions(torch::kShort);
      break;
    case Torch_Int:
      options = torch::TensorOptions(torch::kInt);
      break;
    case Torch_Long:
      options = torch::TensorOptions(torch::kLong);
      break;
    case Torch_Half:
      options = torch::TensorOptions(torch::kHalf);
      break;
    case Torch_Float:
      options = torch::TensorOptions(torch::kFloat);
      break;
    case Torch_Double:
      options = torch::TensorOptions(torch::kDouble);
      break;
    default:
      // TODO handle other types
      break;
  }

  return options;
}

Torch_DataType Torch_ConvertScalarTypeToDataType(torch::ScalarType type) {
  Torch_DataType dtype;
  switch (type) {
    case torch::kByte:
      dtype = Torch_Byte;
      break;
    case torch::kChar:
      dtype = Torch_Char;
      break;
    case torch::kShort:
      dtype = Torch_Short;
      break;
    case torch::kInt:
      dtype = Torch_Int;
      break;
    case torch::kLong:
      dtype = Torch_Long;
      break;
    case torch::kHalf:
      dtype = Torch_Half;
      break;
    case torch::kFloat:
      dtype = Torch_Float;
      break;
    case torch::kDouble:
      dtype = Torch_Double;
      break;
    default:
      dtype = Torch_Unknown;
  }

  return dtype;
}

Torch_IValue Torch_ConvertIValueToTorchIValue(torch::IValue value) {
  if (value.isTensor()) {
    auto tensor = new Torch_Tensor();
    tensor->tensor = value.toTensor();

  if (tensor->tensor.is_cuda()) {
    tensor->tensor = tensor->tensor.to(at::kCPU);
  }
    return Torch_IValue{
        .itype = Torch_IValueTypeTensor,
        .data_ptr = tensor,
    };
  } else if (value.isTuple()) {
    auto elements = value.toTuple()->elements();
    auto tuple = (Torch_IValueTuple*)malloc(sizeof(Torch_IValueTuple));
    auto values = (Torch_IValue*)malloc(sizeof(Torch_IValue) * elements.size());

    for (std::vector<torch::IValue>::size_type i = 0; i != elements.size(); i++) {
      *(values + i) = Torch_ConvertIValueToTorchIValue(elements[i]);
    }

    tuple->values = values;
    tuple->length = elements.size();

    return Torch_IValue{
        .itype = Torch_IValueTypeTuple,
        .data_ptr = tuple,
    };
  }

  return Torch_IValue{};
}

torch::IValue Torch_ConvertTorchIValueToIValue(Torch_IValue value) {
  if (value.itype == Torch_IValueTypeTensor) {
    auto tensor = (Torch_Tensor*)value.data_ptr;
    return tensor->tensor;
  } else if (value.itype == Torch_IValueTypeTuple) {
    auto tuple = (Torch_IValueTuple*)value.data_ptr;
    std::vector<torch::IValue> values;
    values.reserve(tuple->length);

    for (int i = 0; i < tuple->length; i++) {
      auto ival = *(tuple->values + i);
      values.push_back(Torch_ConvertTorchIValueToIValue(ival));
    }

    return torch::jit::Tuple::create(std::move(values));
  }

  // TODO handle this case
  return 0;
}

Torch_TensorContext Torch_NewTensor(void* input_data, int64_t* dimensions, int n_dim, Torch_DataType dtype,
                                    Torch_DeviceKind device) {
  torch::TensorOptions options = Torch_ConvertDataTypeToOptions(dtype);
  std::vector<int64_t> sizes;
  sizes.assign(dimensions, dimensions + n_dim);

  //options = options.device(torch::kCPU, 0);

  torch::Tensor ten = torch::from_blob(input_data, torch::IntArrayRef(sizes), options);

  if (device == CUDA_DEVICE_KIND) {
    ten = ten.to(torch::kCUDA);
  }
  auto tensor = new Torch_Tensor();
  tensor->tensor = ten;

  return (void*)tensor;
}

void* Torch_TensorValue(Torch_TensorContext ctx) {
  auto tensor = reinterpret_cast<Torch_Tensor*>(ctx)->tensor;

  if (tensor.is_cuda()) {
    tensor = tensor.to(at::kCPU);
  }

  return tensor.data_ptr();
}

Torch_DataType Torch_TensorType(Torch_TensorContext ctx) {
  auto tensor = reinterpret_cast<Torch_Tensor*>(ctx)->tensor;
  auto type = tensor.scalar_type();
  return Torch_ConvertScalarTypeToDataType(type);
}

int64_t* Torch_TensorShape(Torch_TensorContext ctx, size_t* dims) {
  auto tensor = reinterpret_cast<Torch_Tensor*>(ctx)->tensor;
  auto sizes = tensor.sizes();
  *dims = sizes.size();
  return (int64_t*)sizes.data();
}

void Torch_PrintTensors(Torch_TensorContext* tensors, size_t input_size) {
  for (int i = 0; i < input_size; i++) {
    auto ctx = tensors + i;
    auto tensor = reinterpret_cast<Torch_Tensor*>(ctx)->tensor;
    std::cout << tensor << "\n";
  }
}

void Torch_DeleteTensor(Torch_TensorContext ctx) {
  auto tensor = (Torch_Tensor*)ctx;
  delete tensor;
}
