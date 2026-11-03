//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#include "sycl-rs-sys/include/queue.hpp"
#include "sycl-rs-sys/include/utils.hpp"
#include "sycl-rs-sys/src/queue-sys.rs.h"

#include <cstring>
#include <stdexcept>

using sycl::ext::intel::property::queue::immediate_command_list;
using sycl::property::queue::in_order;
using sycl_shims::utils::vec_to_vector;

namespace syclexp = sycl::ext::oneapi::experimental;

namespace sycl_shims::queue {
std::unique_ptr<Queue> new_queue() {
  return std::make_unique<Queue>(sycl::queue({in_order()}));
}

std::unique_ptr<Queue> new_queue_immediate() {
  return std::make_unique<Queue>(
      sycl::queue({in_order(), immediate_command_list()}));
}

std::unique_ptr<Queue> new_queue_from_device(Device const &device) {
  return std::make_unique<Queue>(sycl::queue(device, {in_order()}));
}

std::unique_ptr<Queue> new_queue_from_context_and_device(Context const &context,
                                                         Device const &device) {
  return std::make_unique<Queue>(sycl::queue(context, device, {in_order()}));
}

std::unique_ptr<Context> get_context(Queue const &queue) {
  return std::make_unique<Context>(queue.get_context());
}

std::unique_ptr<Device> get_device(Queue const &queue) {
  return std::make_unique<Device>(queue.get_device());
}

std::unique_ptr<Queue> clone(Queue const &queue) {
  return std::make_unique<Queue>(sycl::queue(queue));
}

std::unique_ptr<Event> memset(std::unique_ptr<Queue> &queue, std::uint8_t *ptr,
                              int value, std::size_t num_bytes,
                              rust::Vec<EventPtr> dep_events) {
  return std::make_unique<Event>(queue->memset(
      ptr, value, num_bytes, vec_to_vector(std::move(dep_events))));
}

std::unique_ptr<Event> barrier(std::unique_ptr<Queue> &queue,
                               rust::Vec<EventPtr> dep_events) {
  return std::make_unique<Event>(
      queue->ext_oneapi_submit_barrier(vec_to_vector(std::move(dep_events))));
}

void wait(std::unique_ptr<Queue> &queue) { queue->wait(); }

// Pointers go through the typed set_arg overload rather than raw_kernel_arg: a
// backend may take them through a different entry point than plain bytes (the
// OpenCL CPU device rejects USM pointers passed as bytes), and raw_kernel_arg
// only takes non-pointer values.
static void set_kernel_arg(sycl::handler &cgh, std::size_t index,
                           KernelArg const &arg) {
  switch (arg.kind) {
  case KernelArgKind::Bytes:
    cgh.set_arg(index,
                syclexp::raw_kernel_arg(arg.bytes.data(), arg.bytes.size()));
    return;
  case KernelArgKind::Pointer: {
    void *pointer;
    if (arg.bytes.size() != sizeof(pointer))
      throw std::invalid_argument(
          "pointer kernel argument is not pointer-sized");
    std::memcpy(&pointer, arg.bytes.data(), sizeof(pointer));
    cgh.set_arg(index, pointer);
    return;
  }
  }
  throw std::invalid_argument("unknown kernel argument kind");
}

template <int Dimensions>
std::unique_ptr<Event>
launch(std::unique_ptr<Queue> &queue, sycl::nd_range<Dimensions> nd_range,
       Kernel const &kernel, rust::Slice<KernelArg const> args) {
  return std::make_unique<Event>(queue->submit([&](sycl::handler &cgh) {
    for (std::size_t i = 0; i < args.size(); ++i)
      set_kernel_arg(cgh, i, args[i]);

    cgh.parallel_for(nd_range, kernel);
  }));
}

std::unique_ptr<Event> launch_1d(std::unique_ptr<Queue> &queue,
                                 Range1 global_size, Range1 local_size,
                                 Kernel const &kernel,
                                 rust::Slice<KernelArg const> args) {
  return launch(queue,
                sycl::nd_range<1>{{global_size.data[0]}, {local_size.data[0]}},
                kernel, args);
}

std::unique_ptr<Event> launch_2d(std::unique_ptr<Queue> &queue,
                                 Range2 global_size, Range2 local_size,
                                 Kernel const &kernel,
                                 rust::Slice<KernelArg const> args) {
  return launch(queue,
                sycl::nd_range<2>{{global_size.data[0], global_size.data[1]},
                                  {local_size.data[0], local_size.data[1]}},
                kernel, args);
}

std::unique_ptr<Event> launch_3d(std::unique_ptr<Queue> &queue,
                                 Range3 global_size, Range3 local_size,
                                 Kernel const &kernel,
                                 rust::Slice<KernelArg const> args) {
  return launch(
      queue,
      sycl::nd_range<3>{
          {global_size.data[0], global_size.data[1], global_size.data[2]},
          {local_size.data[0], local_size.data[1], local_size.data[2]}},
      kernel, args);
}

std::unique_ptr<Event> memcpy(std::unique_ptr<Queue> &queue, std::uint8_t *dest,
                              std::uint8_t const *src, std::size_t num_bytes,
                              rust::Vec<EventPtr> dep_events) {
  return std::make_unique<Event>(queue->memcpy(
      dest, src, num_bytes, vec_to_vector(std::move(dep_events))));
}
} // namespace sycl_shims::queue
