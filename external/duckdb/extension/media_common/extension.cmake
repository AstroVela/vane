# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT

function(vane_build_native_media_extension)
  option(VANE_MEDIA_STATIC_DEVELOPMENT_BUILD "Use legacy static media dependencies (requires static release materials)" OFF)
  set(sources native_media_extension.cpp
      ../audio/audio_functions.cpp
      ../image/image_functions.cpp ../image/image_pixel_functions.cpp
      ../image/image_compute_functions.cpp ../image/image_codec.cpp
      ../video/video_functions.cpp ../video/video_frame_functions.cpp ../video/video_index.cpp
      ../media_common/media_reader.cpp ../media_common/image_convert.cpp)
  include_directories(include ../audio/include ../image/include ../video/include
                      ../media_common/include ../file/include)
  find_package(boost_multiprecision CONFIG REQUIRED)

  if(VANE_MEDIA_STATIC_DEVELOPMENT_BUILD)
    find_path(VANE_FFMPEG_CMAKE_DIR FindFFMPEG.cmake PATH_SUFFIXES share/ffmpeg REQUIRED)
    list(APPEND CMAKE_MODULE_PATH "${VANE_FFMPEG_CMAKE_DIR}")
    find_package(FFMPEG REQUIRED)
    find_package(ZLIB REQUIRED)
    find_package(TIFF 4.6.1 REQUIRED)
    find_package(JPEG REQUIRED)
    find_package(WebP CONFIG REQUIRED)
    find_package(SndFile CONFIG REQUIRED)
    find_path(VANE_SOXR_INCLUDE_DIR soxr.h REQUIRED)
    find_library(VANE_SOXR_LIBRARY NAMES soxr REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS} ${VANE_SOXR_INCLUDE_DIR})
    set(dependencies ${FFMPEG_LIBRARIES} TIFF::TIFF JPEG::JPEG ZLIB::ZLIB
                     WebP::webp WebP::webpdemux SndFile::sndfile ${VANE_SOXR_LIBRARY})
  else()
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
      message(FATAL_ERROR "The native media runtime currently supports Linux x86-64 only")
    endif()
    if(NOT EXTENSION_STATIC_BUILD)
      message(FATAL_ERROR "Dynamic media libraries still require EXTENSION_STATIC_BUILD=ON")
    endif()
    if(NOT VANE_MEDIA_RUNTIME_SDK OR NOT EXISTS "${VANE_MEDIA_RUNTIME_DIRECTORY}/runtime-manifest.json")
      message(FATAL_ERROR "native_media requires VANE_MEDIA_RUNTIME_SDK and VANE_MEDIA_RUNTIME_DIRECTORY; see packages/vane-media-runtime/README.md")
    endif()
    include_directories("${VANE_MEDIA_RUNTIME_SDK}/include")
    foreach(library avformat avcodec avutil swscale swresample sndfile soxr tiff jpeg z webp webpdemux)
      set(shared_library "${VANE_MEDIA_RUNTIME_SDK}/lib/lib${library}.so")
      if(NOT EXISTS "${shared_library}")
        message(FATAL_ERROR "Missing native media shared library: ${shared_library}")
      endif()
      list(APPEND dependencies "${shared_library}")
    endforeach()
  endif()

  build_static_extension(native_media ${sources})
  build_loadable_extension(native_media "-warnings" ${sources})
  foreach(target native_media_extension native_media_loadable_extension)
    target_link_libraries(${target} file_extension Boost::multiprecision ${dependencies})
  endforeach()
  if(NOT VANE_MEDIA_STATIC_DEVELOPMENT_BUILD)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set_property(TARGET native_media_loadable_extension PROPERTY
                 VANE_LOADABLE_RUNTIME_DIRECTORY "${VANE_MEDIA_RUNTIME_DIRECTORY}/.libs")
    get_filename_component(vane_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../../.." ABSOLUTE)
    set_property(TARGET native_media_loadable_extension APPEND PROPERTY LINK_DEPENDS
                 "${VANE_MEDIA_RUNTIME_DIRECTORY}/runtime-manifest.json")
    add_custom_command(TARGET native_media_loadable_extension POST_BUILD
      COMMAND "${Python3_EXECUTABLE}" "${vane_root}/scripts/prepare_dynamic_media_extension.py"
              --artifact "$<TARGET_FILE:native_media_loadable_extension>"
              --runtime "${VANE_MEDIA_RUNTIME_DIRECTORY}"
      VERBATIM)
  endif()
endfunction()
