//
// Copyright (C) 2026 Intel Corporation
//
// Under the MIT License or the Apache License v2.0.
// See LICENSE-MIT and LICENSE-APACHE for license information.
// SPDX-License-Identifier: MIT OR Apache-2.0
//

use crate::event::Event;
use crate::info::Info;
use oneapi_rs_sys::event::ffi;

/// Returns the event status of the action associated with this event.
pub struct CommandExecutionStatus;
impl Info for CommandExecutionStatus {
    type Item = crate::info::EventCommandStatus;
    type Target = Event;
    fn get_item(target: &Self::Target) -> Self::Item {
        ffi::get_command_execution_status(&target.0)
    }
}
