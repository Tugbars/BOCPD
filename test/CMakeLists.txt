cmake_minimum_required(VERSION 3.16)
project(bocpd_test_suite C ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

message(STATUS "Building BOCPD Library - Test Suite")

#===============================================================================
# Source Directories
#===============================================================================

set(ASM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../asm)
set(SCALAR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../scalar)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#===============================================================================
# Kernel Variant Selection
#===============================================================================

# BOCPD_KERNEL_VARIANT:
#   0 = Generic (default, works on all AVX2 CPUs)
#   1 = Intel-tuned (Alder Lake, Raptor Lake)
set(BOCPD_KERNEL_VARIANT 0 CACHE STRING "ASM kernel variant (0=generic, 1=intel)")

message(STATUS "Kernel variant: ${BOCPD_KERNEL_VARIANT}")

#===============================================================================
# Compiler Flags
#===============================================================================

if(MSVC)
    set(BOCPD_COMPILE_FLAGS /arch:AVX2 /W4 /O2)
    set(BOCPD_LINK_FLAGS)
else()
    set(BOCPD_COMPILE_FLAGS -march=native -mavx2 -mfma -O3 -Wall -Wextra)
    set(BOCPD_LINK_FLAGS -lm)
endif()

#===============================================================================
# NASM Assembly Support
#===============================================================================

find_program(NASM_EXECUTABLE nasm)

if(NOT NASM_EXECUTABLE)
    message(FATAL_ERROR "NASM not found! Install NASM to build ASM kernels.")
endif()

message(STATUS "Found NASM: ${NASM_EXECUTABLE}")

# Determine object format based on platform
if(WIN32)
    set(NASM_FORMAT win64)
elseif(APPLE)
    set(NASM_FORMAT macho64)
else()
    set(NASM_FORMAT elf64)
endif()

# Custom command to assemble generic kernel
set(ASM_KERNEL_GENERIC_SRC ${ASM_DIR}/bocpd_kernel_avx2_generic.asm)
set(ASM_KERNEL_GENERIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/bocpd_kernel_avx2_generic.o)

add_custom_command(
    OUTPUT ${ASM_KERNEL_GENERIC_OBJ}
    COMMAND ${NASM_EXECUTABLE} -f ${NASM_FORMAT} -o ${ASM_KERNEL_GENERIC_OBJ} ${ASM_KERNEL_GENERIC_SRC}
    DEPENDS ${ASM_KERNEL_GENERIC_SRC}
    COMMENT "Assembling generic AVX2 kernel..."
)

# Custom command to assemble Intel-tuned kernel
set(ASM_KERNEL_INTEL_SRC ${ASM_DIR}/bocpd_kernel_avx2_intel.asm)
set(ASM_KERNEL_INTEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/bocpd_kernel_avx2_intel.o)

add_custom_command(
    OUTPUT ${ASM_KERNEL_INTEL_OBJ}
    COMMAND ${NASM_EXECUTABLE} -f ${NASM_FORMAT} -o ${ASM_KERNEL_INTEL_OBJ} ${ASM_KERNEL_INTEL_SRC}
    DEPENDS ${ASM_KERNEL_INTEL_SRC}
    COMMENT "Assembling Intel-tuned AVX2 kernel..."
)

# Custom target for assembly objects
add_custom_target(asm_kernels DEPENDS 
    ${ASM_KERNEL_GENERIC_OBJ} 
    ${ASM_KERNEL_INTEL_OBJ}
)

#===============================================================================
# Scalar Library (Optimized, No SIMD)
#===============================================================================

add_library(bocpd_scalar STATIC
    ${SCALAR_DIR}/bocpd.c
    ${SCALAR_DIR}/bocpd.h
)

target_include_directories(bocpd_scalar PUBLIC ${SCALAR_DIR})
target_compile_options(bocpd_scalar PRIVATE ${BOCPD_COMPILE_FLAGS})

#===============================================================================
# ASM Library (Ultra-fast AVX2)
#===============================================================================

# Select kernel object based on variant
if(BOCPD_KERNEL_VARIANT EQUAL 1)
    set(ASM_KERNEL_OBJ ${ASM_KERNEL_INTEL_OBJ})
    message(STATUS "Using Intel-tuned kernel")
else()
    set(ASM_KERNEL_OBJ ${ASM_KERNEL_GENERIC_OBJ})
    message(STATUS "Using generic kernel")
endif()

add_library(bocpd_asm STATIC
    ${ASM_DIR}/bocpd_ultra_opt_asm.c
    ${ASM_DIR}/bocpd_asm.h
    ${ASM_KERNEL_OBJ}
)

add_dependencies(bocpd_asm asm_kernels)

target_include_directories(bocpd_asm PUBLIC ${ASM_DIR})
target_compile_options(bocpd_asm PRIVATE ${BOCPD_COMPILE_FLAGS})
target_compile_definitions(bocpd_asm PRIVATE 
    BOCPD_USE_ASM=1
    BOCPD_KERNEL_VARIANT=${BOCPD_KERNEL_VARIANT}
)

#===============================================================================
# Test Executables
#===============================================================================

# Benchmark test (tests both scalar and ASM)
add_executable(bocpd_bench
    ${TEST_DIR}/bocpd_bench.c
)

target_link_libraries(bocpd_bench PRIVATE bocpd_scalar bocpd_asm)
target_include_directories(bocpd_bench PRIVATE ${SCALAR_DIR} ${ASM_DIR})
target_compile_options(bocpd_bench PRIVATE ${BOCPD_COMPILE_FLAGS})
target_compile_definitions(bocpd_bench PRIVATE STANDALONE)

if(WIN32 AND MINGW)
    target_link_options(bocpd_bench PRIVATE -mconsole)
endif()

if(UNIX OR MINGW)
    target_link_libraries(bocpd_bench PRIVATE m)
endif()

# Fast implementation test (ASM version)
add_executable(test_bocpd_fast
    ${TEST_DIR}/test_bocpd_fast.c
)

target_link_libraries(test_bocpd_fast PRIVATE bocpd_asm)
target_include_directories(test_bocpd_fast PRIVATE ${ASM_DIR})
target_compile_options(test_bocpd_fast PRIVATE ${BOCPD_COMPILE_FLAGS})
target_compile_definitions(test_bocpd_fast PRIVATE STANDALONE)

if(WIN32 AND MINGW)
    target_link_options(test_bocpd_fast PRIVATE -mconsole)
endif()

if(UNIX OR MINGW)
    target_link_libraries(test_bocpd_fast PRIVATE m)
endif()

# Scalar test (standalone scalar version test)
add_executable(test_bocpd_scalar
    ${TEST_DIR}/test_bocpd_scalar.c
)

target_link_libraries(test_bocpd_scalar PRIVATE bocpd_scalar)
target_include_directories(test_bocpd_scalar PRIVATE ${SCALAR_DIR})
target_compile_options(test_bocpd_scalar PRIVATE ${BOCPD_COMPILE_FLAGS})
target_compile_definitions(test_bocpd_scalar PRIVATE STANDALONE)

if(WIN32 AND MINGW)
    target_link_options(test_bocpd_scalar PRIVATE -mconsole)
endif()

if(UNIX OR MINGW)
    target_link_libraries(test_bocpd_scalar PRIVATE m)
endif()

#===============================================================================
# CTest Integration
#===============================================================================

enable_testing()

add_test(NAME Benchmark COMMAND bocpd_bench)
add_test(NAME FastASM COMMAND test_bocpd_fast)
add_test(NAME Scalar COMMAND test_bocpd_scalar)

set_tests_properties(Benchmark PROPERTIES TIMEOUT 120)
set_tests_properties(FastASM PROPERTIES TIMEOUT 60)
set_tests_properties(Scalar PROPERTIES TIMEOUT 60)

#===============================================================================
# Custom Targets
#===============================================================================

# Run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS bocpd_bench test_bocpd_fast test_bocpd_scalar
    COMMENT "Running all BOCPD tests..."
)

# Run benchmark only
add_custom_target(run_bench
    COMMAND bocpd_bench
    DEPENDS bocpd_bench
    COMMENT "Running BOCPD benchmark (scalar vs ASM)..."
)

# Run ASM tests
add_custom_target(run_fast
    COMMAND test_bocpd_fast
    DEPENDS test_bocpd_fast
    COMMENT "Running ASM kernel tests..."
)

# Run scalar tests
add_custom_target(run_scalar
    COMMAND test_bocpd_scalar
    DEPENDS test_bocpd_scalar
    COMMENT "Running scalar implementation tests..."
)

#===============================================================================
# Address Sanitizer Option
#===============================================================================

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(ENABLE_ASAN)
    if(NOT MSVC)
        set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer -g)
        
        target_compile_options(test_bocpd_fast PRIVATE ${ASAN_FLAGS})
        target_link_options(test_bocpd_fast PRIVATE ${ASAN_FLAGS})
        
        target_compile_options(test_bocpd_scalar PRIVATE ${ASAN_FLAGS})
        target_link_options(test_bocpd_scalar PRIVATE ${ASAN_FLAGS})
        
        message(STATUS "AddressSanitizer ENABLED")
    else()
        target_compile_options(test_bocpd_fast PRIVATE /fsanitize=address)
        target_compile_options(test_bocpd_scalar PRIVATE /fsanitize=address)
        message(STATUS "AddressSanitizer ENABLED (MSVC)")
    endif()
endif()

#===============================================================================
# Build Summary
#===============================================================================

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════════╗")
message(STATUS "║              BOCPD Test Suite Configuration                ║")
message(STATUS "╚════════════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Directories:")
message(STATUS "  ASM sources:     ${ASM_DIR}")
message(STATUS "  Scalar sources:  ${SCALAR_DIR}")
message(STATUS "  Tests:           ${TEST_DIR}")
message(STATUS "")
message(STATUS "Build configuration:")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Kernel variant:   ${BOCPD_KERNEL_VARIANT} (0=generic, 1=intel)")
message(STATUS "  NASM format:      ${NASM_FORMAT}")
message(STATUS "  AddressSanitizer: ${ENABLE_ASAN}")
message(STATUS "")
message(STATUS "Libraries:")
message(STATUS "  bocpd_scalar     - Optimized scalar (no SIMD)")
message(STATUS "  bocpd_asm        - Ultra-fast AVX2 + ASM kernels")
message(STATUS "")
message(STATUS "Test targets:")
message(STATUS "  bocpd_bench      - Benchmark (scalar vs ASM)")
message(STATUS "  test_bocpd_fast  - ASM kernel tests")
message(STATUS "  test_bocpd_scalar - Scalar implementation tests")
message(STATUS "")
message(STATUS "Custom targets (make <target>):")
message(STATUS "  run_tests        - Run all tests via CTest")
message(STATUS "  run_bench        - Run benchmark only")
message(STATUS "  run_fast         - Run ASM tests only")
message(STATUS "  run_scalar       - Run scalar tests only")
message(STATUS "")
message(STATUS "Build with Intel-tuned kernel:")
message(STATUS "  cmake -DBOCPD_KERNEL_VARIANT=1 ..")
message(STATUS "")