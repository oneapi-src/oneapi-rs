//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

#pragma once

#include <memory>

#include "oneapi-rs-sys/include/types.hpp"
#include "rust/cxx.h"

namespace sycl_shims {
struct EventPtr;
struct Range1;
struct Range2;
struct Range3;
} // namespace sycl_shims

namespace sycl_shims::queue {
std::unique_ptr<Queue> new_queue();
std::unique_ptr<Queue> new_queue_immediate();
std::unique_ptr<Queue> new_queue_from_device(Device const &);
std::unique_ptr<Context> get_context(Queue const &);
std::unique_ptr<Queue> clone(Queue const &);
std::unique_ptr<Event> memset(std::unique_ptr<Queue> &, std::uint8_t *ptr,
                              int value, std::size_t num_bytes,
                              rust::Vec<EventPtr>);
std::unique_ptr<Event> barrier(std::unique_ptr<Queue> &, rust::Vec<EventPtr>);
void wait(std::unique_ptr<Queue> &);

std::unique_ptr<Event>
launch_1d(std::unique_ptr<Queue> &, Range1 global_size, Range1 local_size,
          Kernel const &,
          rust::Slice<rust::slice<std::uint8_t const> const> args);

std::unique_ptr<Event>
launch_2d(std::unique_ptr<Queue> &, Range2 global_size, Range2 local_size,
          Kernel const &,
          rust::Slice<rust::slice<std::uint8_t const> const> args);

std::unique_ptr<Event>
launch_3d(std::unique_ptr<Queue> &, Range3 global_size, Range3 local_size,
          Kernel const &,
          rust::Slice<rust::slice<std::uint8_t const> const> args);
} // namespace sycl_shims::queue
