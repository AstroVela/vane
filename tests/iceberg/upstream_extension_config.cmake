# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: Apache-2.0

# Register the production-pinned Iceberg extension with its upstream SQL tests
# enabled. The release configuration deliberately keeps automatic test loading
# disabled, so the compatibility runner uses this test-only configuration.
set(_VANE_ICEBERG_PRODUCTION_CONFIG
    "${CMAKE_CURRENT_LIST_DIR}/../../external/duckdb/.github/config/extensions/iceberg.cmake"
)
if(NOT EXISTS "${_VANE_ICEBERG_PRODUCTION_CONFIG}")
  message(
    FATAL_ERROR
      "Pinned Iceberg extension configuration not found: ${_VANE_ICEBERG_PRODUCTION_CONFIG}"
  )
endif()

file(READ "${_VANE_ICEBERG_PRODUCTION_CONFIG}"
     _VANE_ICEBERG_PRODUCTION_CONFIG_CONTENT)
string(REGEX MATCH "GIT_URL[ \t]+([^ \t\r\n]+)" _VANE_ICEBERG_GIT_URL_MATCH
             "${_VANE_ICEBERG_PRODUCTION_CONFIG_CONTENT}")
set(_VANE_ICEBERG_GIT_URL "${CMAKE_MATCH_1}")
string(REGEX MATCH "GIT_TAG[ \t]+([0-9a-f]+)" _VANE_ICEBERG_GIT_TAG_MATCH
             "${_VANE_ICEBERG_PRODUCTION_CONFIG_CONTENT}")
set(_VANE_ICEBERG_GIT_TAG "${CMAKE_MATCH_1}")
string(LENGTH "${_VANE_ICEBERG_GIT_TAG}" _VANE_ICEBERG_GIT_TAG_LENGTH)

if(NOT _VANE_ICEBERG_GIT_URL OR NOT _VANE_ICEBERG_GIT_TAG_LENGTH EQUAL 40)
  message(
    FATAL_ERROR
      "Could not resolve the pinned Iceberg URL and 40-character revision from ${_VANE_ICEBERG_PRODUCTION_CONFIG}"
  )
endif()

duckdb_extension_load(
  iceberg
  LOAD_TESTS
  GIT_URL
  "${_VANE_ICEBERG_GIT_URL}"
  GIT_TAG
  "${_VANE_ICEBERG_GIT_TAG}"
  APPLY_PATCHES)

unset(_VANE_ICEBERG_GIT_TAG_LENGTH)
unset(_VANE_ICEBERG_GIT_TAG)
unset(_VANE_ICEBERG_GIT_TAG_MATCH)
unset(_VANE_ICEBERG_GIT_URL)
unset(_VANE_ICEBERG_GIT_URL_MATCH)
unset(_VANE_ICEBERG_PRODUCTION_CONFIG_CONTENT)
unset(_VANE_ICEBERG_PRODUCTION_CONFIG)
