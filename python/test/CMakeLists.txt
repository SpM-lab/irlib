file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION ${CMAKE_BINARY_DIR}/python/test FILES_MATCHING PATTERN "*.py")

add_python_test(basis_test)
