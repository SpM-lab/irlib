function(add_gtest test)
    if (TestXMLOutput)
        set (test_xml_output --gtest_output=xml:${test}.xml)
    endif(TestXMLOutput)

    set(source "${ARGV1}/${test}.cpp;${ARGV1}/common.cpp")
    set(gtest_src "${ARGV1}/gtest_main.cc;${ARGV1}/gtest-all.cc")

    add_executable(${test} ${source} ${gtest_src} ${header_files})
    target_link_libraries(${test} ${LINK_ALL})
    add_test(NAME ${test} COMMAND ${test} ${test_xml_output})
endfunction(add_gtest)

#include_directories(${IRBASIS_LIBRARY_INCLUDE_DIRS})
list(APPEND LINK_ALL ${Boost_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}" CACHE STRING "Linker flags for executables" FORCE)

install(DIRECTORY "include" DESTINATION "." FILES_MATCHING PATTERN "*.hpp")

file(GLOB_RECURSE header_files *.hpp)

if (Testing)
  #testing source files
  set(unittest_src unittest mpmath kernel)
  foreach(test ${unittest_src})
      add_gtest(${test} test)
  endforeach(test)
endif()
