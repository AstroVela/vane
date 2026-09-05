# SPDX-FileCopyrightText: 2026 Vane contributors SPDX-License-Identifier: MIT

function(vane_build_media_extension domain)
  # vcpkg's FFmpeg package supplies FindFFMPEG rather than a config package.
  find_path(VANE_FFMPEG_CMAKE_DIR FindFFMPEG.cmake PATH_SUFFIXES share/ffmpeg
                                                                 REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${VANE_FFMPEG_CMAKE_DIR}")
  find_package(FFMPEG REQUIRED)
  set(components libavformat libavcodec libavutil)
  set(sources ${domain}_extension.cpp ${domain}_functions.cpp
              ../media_common/media_reader.cpp)
  if(domain STREQUAL "audio")
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
  endforeach()
endfunction()
