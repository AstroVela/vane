# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT

function(vane_build_media_extension domain)
  # vcpkg's FFmpeg package supplies FindFFMPEG rather than a config package.
  find_path(VANE_FFMPEG_CMAKE_DIR FindFFMPEG.cmake PATH_SUFFIXES share/ffmpeg
                                                                 REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${VANE_FFMPEG_CMAKE_DIR}")
  find_package(FFMPEG REQUIRED)
  set(components libavformat libavcodec libavutil)
  set(sources ${domain}_extension.cpp ${domain}_functions.cpp
              ../media_common/media_reader.cpp)
  if(domain STREQUAL "image")
    find_package(ZLIB REQUIRED)
    find_package(TIFF 4.6.1 REQUIRED)
    find_package(JPEG REQUIRED)
    find_package(WebP CONFIG REQUIRED)
    list(APPEND sources image_pixel_functions.cpp image_compute_functions.cpp
         image_codec.cpp)
  endif()
  if(domain STREQUAL "video")
    find_package(boost_multiprecision CONFIG REQUIRED)
    list(APPEND sources video_frame_functions.cpp video_index.cpp)
  endif()
  if(domain STREQUAL "audio")
    find_package(SndFile CONFIG REQUIRED)
    find_path(VANE_SOXR_INCLUDE_DIR soxr.h REQUIRED)
    find_library(VANE_SOXR_LIBRARY NAMES soxr REQUIRED)
    list(APPEND components libswresample)
  else()
    list(APPEND components libswscale)
    list(APPEND sources ../media_common/image_convert.cpp)
  endif()
  foreach(component IN LISTS components)
    if(NOT FFMPEG_${component}_LIBRARY)
      message(
        FATAL_ERROR "The ${domain} extension requires FFmpeg ${component}")
    endif()
  endforeach()
  include_directories(include ../media_common/include ../file/include
                      ${FFMPEG_INCLUDE_DIRS})
  build_static_extension(${domain} ${sources})
  build_loadable_extension(${domain} "-warnings" ${sources})
  foreach(target IN ITEMS ${domain}_extension ${domain}_loadable_extension)
    target_link_libraries(${target} file_extension ${FFMPEG_LIBRARIES})
    if(domain STREQUAL "image")
      target_link_libraries(${target} TIFF::TIFF JPEG::JPEG ZLIB::ZLIB
                            WebP::webp WebP::webpdemux)
    endif()
    if(domain STREQUAL "audio")
      target_include_directories(${target} PRIVATE ${VANE_SOXR_INCLUDE_DIR})
      target_link_libraries(${target} SndFile::sndfile ${VANE_SOXR_LIBRARY})
    endif()
    if(domain STREQUAL "video")
      target_link_libraries(${target} Boost::multiprecision)
    endif()
  endforeach()
endfunction()
