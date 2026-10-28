# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT

# MSBuild runs POST_BUILD events even when the linker has no work. Relocation
# rewrites the binary, so run it once per changed artifact/runtime input.
set(needs_preparation FALSE)
if(NOT EXISTS "${STAMP}")
  set(needs_preparation TRUE)
endif()
foreach(input IN LISTS INPUTS)
  if(NOT EXISTS "${input}" OR "${input}" IS_NEWER_THAN "${STAMP}")
    set(needs_preparation TRUE)
  endif()
endforeach()
if(needs_preparation)
  execute_process(COMMAND "${PYTHON}" "${PREPARE}" --artifact "${ARTIFACT}"
                          --runtime "${RUNTIME}" COMMAND_ERROR_IS_FATAL ANY)
  file(TOUCH "${STAMP}")
endif()
