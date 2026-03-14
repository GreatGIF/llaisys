#include "mx_resource.hpp"

namespace llaisys::device::mx {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {}

} // namespace llaisys::device::mx
