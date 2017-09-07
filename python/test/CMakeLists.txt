file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION ${CMAKE_BINARY_DIR}/python/test FILES_MATCHING PATTERN "*.py")

add_test(NAME basis_test COMMAND ${PYTHON_EXECUTABLE} basis_test.py)
#add_python_test(babsis_test)
