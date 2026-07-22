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
#include "oneapi-rs-sys/src/event-sys.rs.h"
#include "rust/cxx.h"

namespace sycl_shims {
enum class EventCommandStatus : std::uint8_t;
} // namespace sycl_shims

namespace sycl_shims::event {
void wait(std::unique_ptr<Event> &);
void register_callback(std::unique_ptr<Queue> &, Event const &,
                       SharedWaker const *);
EventCommandStatus get_command_execution_status(Event const &);
std::unique_ptr<Event> clone(Event const &);
} // namespace sycl_shims::event
