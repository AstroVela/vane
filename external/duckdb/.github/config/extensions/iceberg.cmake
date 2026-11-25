# Windows tests for iceberg currently not working
IF (NOT WIN32)
  set(LOAD_ICEBERG_TESTS "LOAD_TESTS")
else ()
  set(LOAD_ICEBERG_TESTS "")
endif()
if (NOT MINGW)
  set(ICEBERG_VANE_NATIVE ON CACHE BOOL
      "Build Iceberg against the Vane native engine interfaces" FORCE)
  duckdb_extension_load(iceberg
     #FIXME: restore autoloading tests ${LOAD_ICEBERG_TESTS}
            GIT_URL https://github.com/AstroVela/duckdb-iceberg
            GIT_TAG f83b867b9b4c5b4a1c7eefd61a16e46a8cf55b1c
            )
endif()
