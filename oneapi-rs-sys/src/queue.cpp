//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#include "oneapi-rs-sys/include/queue.hpp"
#include "oneapi-rs-sys/src/queue-sys.rs.h"

using sycl::property::queue::in_order;
using sycl::ext::intel::property::queue::immediate_command_list;

namespace sycl_shims::queue {
std::unique_ptr<Queue> new_queue() {
  return std::make_unique<Queue>(sycl::queue({
    in_order()
  }));
}
std::unique_ptr<Queue> new_queue_immediate() {
  return std::make_unique<Queue>(sycl::queue({
    in_order(),
    immediate_command_list()
  }));
}
std::unique_ptr<Queue> new_queue_from_device(Device const & device) {
  return std::make_unique<Queue>(sycl::queue(device, {in_order()}));
}
std::unique_ptr<Queue> clone(Queue const & queue) {
  return std::make_unique<Queue>(sycl::queue(queue));
}
std::unique_ptr<Event> memset(
  std::unique_ptr<Queue> & queue,
  std::uint8_t * ptr,
  int value,
  std::size_t num_bytes,
  rust::Vec<EventPtr> dep_events
) {
  std::vector<sycl::event> deps;
  for (auto&& e: dep_events)
    deps.push_back(std::move(*e.ptr));
  return std::make_unique<Event>(queue->memset(ptr, value, num_bytes, deps));
}
std::unique_ptr<Event> barrier(std::unique_ptr<Queue> & queue, rust::Vec<EventPtr> dep_events) {
  std::vector<sycl::event> deps;
  for (auto&& e: dep_events)
    deps.push_back(std::move(*e.ptr));
  return std::make_unique<Event>(queue->ext_oneapi_submit_barrier(deps));
}
void wait(std::unique_ptr<Queue> & queue) {
  queue->wait();
}
} // namespace sycl_shims::queue
