//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#include "oneapi-rs-sys/include/queue.hpp"
#include "oneapi-rs-sys/src/queue-sys.rs.h"

using sycl::ext::intel::property::queue::immediate_command_list;
using sycl::property::queue::in_order;

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

std::unique_ptr<Context> get_context(Queue const &queue) {
  return std::make_unique<Context>(queue.get_context());
}

std::unique_ptr<Queue> clone(Queue const &queue) {
  return std::make_unique<Queue>(sycl::queue(queue));
}

std::unique_ptr<Event> memset(std::unique_ptr<Queue> &queue, std::uint8_t *ptr,
                              int value, std::size_t num_bytes,
                              rust::Vec<EventPtr> dep_events) {
  std::vector<sycl::event> deps;
  for (auto &&e : dep_events)
    deps.push_back(std::move(*e.ptr));
  return std::make_unique<Event>(queue->memset(ptr, value, num_bytes, deps));
}

std::unique_ptr<Event> barrier(std::unique_ptr<Queue> &queue,
                               rust::Vec<EventPtr> dep_events) {
  std::vector<sycl::event> deps;
  for (auto &&e : dep_events)
    deps.push_back(std::move(*e.ptr));
  return std::make_unique<Event>(queue->ext_oneapi_submit_barrier(deps));
}

void wait(std::unique_ptr<Queue> &queue) { queue->wait(); }

template <int Dimensions>
std::unique_ptr<Event>
launch(std::unique_ptr<Queue> &queue, sycl::nd_range<Dimensions> nd_range,
       Kernel const &kernel,
       rust::Slice<rust::slice<std::uint8_t const> const> args) {
  return std::make_unique<Event>(queue->submit([&](sycl::handler &cgh) {
    for (std::size_t i = 0; i < args.size(); ++i)
      cgh.set_arg(i, syclexp::raw_kernel_arg(args[i].data(), args[i].size()));

    cgh.parallel_for(nd_range, kernel);
  }));
}

std::unique_ptr<Event>
launch_1d(std::unique_ptr<Queue> &queue, Range1 global_size, Range1 local_size,
          Kernel const &kernel,
          rust::Slice<rust::slice<std::uint8_t const> const> args) {
  return launch(queue,
                sycl::nd_range<1>{{global_size.data[0]}, {local_size.data[0]}},
                kernel, args);
}

std::unique_ptr<Event>
launch_2d(std::unique_ptr<Queue> &queue, Range2 global_size, Range2 local_size,
          Kernel const &kernel,
          rust::Slice<rust::slice<std::uint8_t const> const> args) {
  return launch(queue,
                sycl::nd_range<2>{{global_size.data[0], global_size.data[1]},
                                  {local_size.data[0], local_size.data[1]}},
                kernel, args);
}

std::unique_ptr<Event>
launch_3d(std::unique_ptr<Queue> &queue, Range3 global_size, Range3 local_size,
          Kernel const &kernel,
          rust::Slice<rust::slice<std::uint8_t const> const> args) {
  return launch(
      queue,
      sycl::nd_range<3>{
          {global_size.data[0], global_size.data[1], global_size.data[2]},
          {local_size.data[0], local_size.data[1], local_size.data[2]}},
      kernel, args);
}
} // namespace sycl_shims::queue
