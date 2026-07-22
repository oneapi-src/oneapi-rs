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
} // namespace sycl_shims

namespace sycl_shims::queue {
std::unique_ptr<Queue> new_queue();
std::unique_ptr<Queue> new_queue_immediate();
std::unique_ptr<Queue> new_queue_from_device(Device const &);
std::unique_ptr<Queue> clone(Queue const &);
std::unique_ptr<Event> memset(std::unique_ptr<Queue> &, std::uint8_t *ptr,
                              int value, std::size_t num_bytes,
                              rust::Vec<EventPtr>);
std::unique_ptr<Event> barrier(std::unique_ptr<Queue> &, rust::Vec<EventPtr>);
void wait(std::unique_ptr<Queue> &);
} // namespace sycl_shims::queue
