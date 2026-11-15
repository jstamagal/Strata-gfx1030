# cmake/backends/hip.cmake - the AMD (HIP) parts of the root project: the gfx906 compat toolchain and sources, MMQ on HIP.
#
# Every block below is a macro: it runs in the scope of the caller (CMakeLists.txt calls it where the block used to be), so variables, targets and
# their order are those of the single-file CMakeLists.txt it was cut from (cleanup 5.4; the build graph is identical).

# HIP for gfx906 through the CUDA compat layer: the toolchain and the CUDA:: stand-in targets
macro(strata_hip_gfx906_toolchain)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx906 CACHE STRING "HIP architectures" FORCE)
  endif()
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  add_library(CUDA::cudart INTERFACE IMPORTED)
  target_link_libraries(CUDA::cudart INTERFACE hip::host)
  target_compile_definitions(CUDA::cudart INTERFACE STRATA_HIP_GFX906=1 __HIP_PLATFORM_AMD__=1)
  target_include_directories(CUDA::cudart BEFORE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  add_library(CUDA::cublas INTERFACE IMPORTED)
  target_link_libraries(CUDA::cublas INTERFACE roc::hipblas CUDA::cudart)
  set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/platform/hip_compat)
  add_compile_definitions(STRATA_HIP_GFX906=1)
  message(STATUS "Strata: HIP (gfx906 compat) enabled, arch ${CMAKE_HIP_ARCHITECTURES}")
  set(_strata_gpu_runtime_target CUDA::cudart)
  set(_strata_gpu_blas_target CUDA::cublas)
  set(_strata_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endmacro()

# strata_mmq on HIP (gfx11 / gfx12 / gfx906): the same MMQ sources compiled as HIP, and their tests
macro(strata_hip_mmq)
    # Compile only the pinned GGML MMQ and activation-quantizer sources used by prefill. This uses
    # common.cuh's native HIP path without enabling a second GGML HIP backend.
    set(_strata_gg "${_strata_ggml_root}/ggml")
    set(_strata_mmq "${_strata_gg}/src/ggml-cuda")
    set(_strata_quantize_cu "${_strata_mmq}/quantize.cu")   # the overlay's copy when the trellis overlay is on: it includes mmq.cuh
    if(_strata_mmq_tq_overlay)
      set(_strata_quantize_cu "${_strata_tq_overlay_dir}/quantize.cu")
    endif()
    if(NOT EXISTS "${_strata_mmq}/common.cuh" OR NOT EXISTS "${_strata_mmq}/quantize.cu")
      message(FATAL_ERROR "The selected GGML source is missing the ggml-cuda MMQ sources: ${_strata_mmq}")
    endif()
    # Aurora: moe_fused.cu's gfx11 and moe_fused_iq.cu's gfx11 / gfx12 WMMA kernels for the prompt experts (opt-in STRATA_PF_FUSED=1)
    set(_strata_mmq_sources
      src/prefill/moe_mmq.cu
      src/prefill/moe_fused.cu
      src/prefill/moe_fused_iq.cu
      src/prefill/moe_q4.cu
      src/prefill/moe_f16_kq.cu
      src/prefill/ggml_cuda_host.cu
      "${_strata_quantize_cu}")
    if(STRATA_MMQ_KQUANTS)
      # the same K-quant instances as the CUDA branch: on HIP they serve the dense GGUF projections
      # of Gemm::native (STRATA_DENSE_MMQ=1), the mixed-quant packs' attention/PLE tensors
      list(APPEND _strata_mmq_types q4_k q5_k q5_1 q6_k)
    endif()
    # Unsloth's UD-Q3_K_XL MoE files (Ornith 1.5 122B) mix Q3_K downs and IQ1_S experts into a few layers: without
    # these two the whole prompt path falls back to the native grouped kernels.  HIP only (no CUDA VRAM cost).
    list(APPEND _strata_mmq_types q3_k iq1_s)
    foreach(_t IN LISTS _strata_mmq_types)
      list(APPEND _strata_mmq_sources "${_strata_mmq}/template-instances/mmq-instance-${_t}.cu")
    endforeach()
    if(_strata_mmq_tq_overlay)
      foreach(_t tq2_t tqk6 tqk7 tqp5 tqp6 tqp7)
        list(APPEND _strata_mmq_sources "${_strata_tq_overlay_dir}/template-instances/mmq-instance-${_t}.cu")
      endforeach()
    endif()
    set_source_files_properties(${_strata_mmq_sources} PROPERTIES LANGUAGE HIP)
    if(STRATA_MMQ_KQUANTS AND STRATA_MMQ_WIDE_LDS)
      # The wide-LDS activation windows of the K-quant MMQ kernels (llama.cpp-rdna2's, third_party/llama.cpp-patches)
      # are compiled only into a template instance whose source file carries the macro of its type; ggml's own HIP
      # CMake sets them, Strata's build does not use that file.  Only the RDNA2 macros: mmq.cuh reads them in the
      # gfx103x device code alone, so another architecture of the same build keeps the stock loads.
      file(STRINGS "${_strata_mmq}/mmq.cuh" _strata_wide_lds REGEX "GGML_HIP_RDNA2_Q4_K_MMQ_WIDE_LDS")
      if(_strata_wide_lds)
        foreach(_t q4_k q5_k q6_k)
          string(TOUPPER "${_t}" _tu)
          set_source_files_properties("${_strata_mmq}/template-instances/mmq-instance-${_t}.cu"
                                      PROPERTIES COMPILE_DEFINITIONS "GGML_HIP_RDNA2_${_tu}_MMQ_WIDE_LDS=1")
        endforeach()
        message(STATUS "Strata: K-quant MMQ wide-LDS activation windows on for gfx103x (STRATA_MMQ_WIDE_LDS)")
      else()
        message(STATUS "Strata: K-quant MMQ wide-LDS activation windows off: ${_strata_mmq}/mmq.cuh has no such path")
      endif()
    endif()
    # moe_fused.cu is Strata's own CUDA-shaped source: it takes the HIP compat headers, ggml's MMQ sources do not
    set_source_files_properties(src/prefill/moe_fused.cu src/prefill/moe_fused_iq.cu src/prefill/moe_q4.cu src/prefill/moe_f16_kq.cu
                                PROPERTIES INCLUDE_DIRECTORIES "${STRATA_HIP_COMPAT_INCLUDE_DIR}")
    add_library(strata_mmq STATIC ${_strata_mmq_sources})
    target_include_directories(strata_mmq
      PRIVATE "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}"
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(_strata_mmq_tq_overlay)
      target_include_directories(strata_mmq BEFORE PRIVATE ${_strata_tq_overlay_dir} ${_strata_tq_overlay_dir}/include)
    endif()
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP=1 STRATA_MMQ_Q3K_IQ1S=1)
    if(STRATA_MMQ_VEC_COPY)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_VEC_COPY=1)
    endif()
    if(NOT STRATA_MMQ_RDNA2_TILES)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_RDNA2_TILES_OFF=1)
    endif()
    if(NOT STRATA_MMQ_RDNA2_DOT_Z)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_RDNA2_DOT_Z_OFF=1)
    endif()
    file(STRINGS "${_strata_mmq}/mmq.cuh" _strata_q8k_path REGEX "mmq_use_q8_K_activations")
    file(STRINGS "${_strata_mmq}/quantize.cu" _strata_iq_q8k_bridge REGEX "strata_quantize_mmq_iq_q8k_cuda")
    if(STRATA_MMQ_RDNA2_Q8K AND STRATA_MMQ_KQUANTS AND _strata_q8k_path AND _strata_iq_q8k_bridge)
      target_compile_definitions(strata_mmq PRIVATE GGML_CUDA_MMQ_Q8K=1 STRATA_MMQ_Q8K_BUILT=1)
      message(STATUS "Strata: owner RDNA2 Q8_K MMQ experiment built; runtime STRATA_MMQ_Q8K=1 (not bit-exact)")
    endif()
    if(STRATA_MMQ_KQUANTS)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_KQUANTS=1)
    endif()
    if(STRATA_ORCA_Q4KS_MMQ)
      target_compile_definitions(strata_mmq PRIVATE STRATA_ORCA_Q4KS_MMQ=1)
    endif()
    if(_strata_mmq_tq)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQ=1)
    endif()
    if(_strata_mmq_tqp)
      target_compile_definitions(strata_mmq PRIVATE STRATA_MMQ_TQP=1)
    endif()
    target_link_libraries(strata_mmq PUBLIC ggml-base hip::host roc::hipblas)
    target_link_libraries(strata_prefill PUBLIC strata_mmq)
    target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1)
    # the RDNA2 Q4_0 / Q4_1 prompt-expert kernel (moe_q4.cu) against an FP64 reference of the same arithmetic, and
    # its timing on the two real shapes (exit 77 off gfx103x)
    add_executable(hip_prefill_q4_parity tests/hip/prefill_q4_parity.cpp)
    target_include_directories(hip_prefill_q4_parity PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_q4_parity PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_q4_parity COMMAND hip_prefill_q4_parity)
    set_tests_properties(hip_prefill_q4_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    list(APPEND _strata_hip_tests hip_prefill_q4_parity)
    # the experts' two products for every weight type of the gfx103x packs: an FP64 check, an output hash and a
    # timing in G MAC/s (a tool for an A/B of an expert kernel; not a ctest)
    add_executable(hip_prefill_expert_bench tests/hip/prefill_expert_bench.cpp)
    target_include_directories(hip_prefill_expert_bench PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_expert_bench PRIVATE strata_mmq ggml-base strata_hip_runtime)
    # Fresh MMQ outputs, then an all-output comparison on the same q8_1 inputs.
    # Fixtures order the processes: each environment switch is cached.
    foreach(_iq_type IN ITEMS 22 18 21)
        add_test(NAME hip_prefill_iq${_iq_type}_reference COMMAND ${CMAKE_COMMAND} -E env
            STRATA_PF_IQ2S=0 STRATA_PF_IQ3=0 STRATA_PF_DIRECT=1 $<TARGET_FILE:hip_prefill_expert_bench>
            --type ${_iq_type} --rows 40 --write-reference ${CMAKE_CURRENT_BINARY_DIR}/iq${_iq_type}-reference)
        add_test(NAME hip_prefill_iq${_iq_type}_parity COMMAND ${CMAKE_COMMAND} -E env
            STRATA_PF_IQ2S=1 STRATA_PF_IQ3=1 STRATA_PF_DIRECT=1 $<TARGET_FILE:hip_prefill_expert_bench>
            --type ${_iq_type} --rows 40 --check-reference ${CMAKE_CURRENT_BINARY_DIR}/iq${_iq_type}-reference)
        set_tests_properties(hip_prefill_iq${_iq_type}_reference PROPERTIES
            FIXTURES_SETUP iq${_iq_type}_reference TIMEOUT 300 SKIP_RETURN_CODE 77)
        set_tests_properties(hip_prefill_iq${_iq_type}_parity PROPERTIES
            FIXTURES_REQUIRED iq${_iq_type}_reference TIMEOUT 300 SKIP_RETURN_CODE 77)
        list(APPEND _strata_hip_tests hip_prefill_iq${_iq_type}_reference hip_prefill_iq${_iq_type}_parity)
    endforeach()

    if(STRATA_MMQ_KQUANTS)
      # llama.cpp's Q4_K / Q5_K / Q6_K MMQ kernels through mmq::quantize / mmq::Context::run (the experts' gate+up
      # shape, a dense projection, uneven experts) against an FP64 reference of the same arithmetic; prints a hash
      # of every output, so two builds compare bit for bit, and times each product
      add_executable(hip_prefill_kquant_parity tests/hip/prefill_kquant_parity.cpp)
      target_include_directories(hip_prefill_kquant_parity PRIVATE "${_strata_gg}/include")
      target_link_libraries(hip_prefill_kquant_parity PRIVATE strata_mmq ggml-base strata_hip_runtime)
      add_test(NAME hip_prefill_kquant_parity COMMAND hip_prefill_kquant_parity)
      set_tests_properties(hip_prefill_kquant_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
      list(APPEND _strata_hip_tests hip_prefill_kquant_parity)
    endif()
    # Aurora: the gfx11 fused experts against the FP64 reference and MMQ (tests/cuda's test; exit 77 off gfx11)
    add_executable(hip_prefill_fused_moe tests/cuda/prefill_fused_moe_test.cpp)
    target_include_directories(hip_prefill_fused_moe PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_moe PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_moe COMMAND hip_prefill_fused_moe)
    set_tests_properties(hip_prefill_fused_moe PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    list(APPEND _strata_hip_tests hip_prefill_fused_moe)
    add_executable(hip_prefill_fused_iq tests/cuda/prefill_fused_iq_test.cpp)
    target_include_directories(hip_prefill_fused_iq PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_fused_iq PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_fused_iq COMMAND hip_prefill_fused_iq)
    set_tests_properties(hip_prefill_fused_iq PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    # the same binary on F16 weights (gfx11: the covered formats take moe_f16_kq.cu / native_f16_kernel; elsewhere it is the default run again)
    add_test(NAME hip_prefill_fused_iq_f16 COMMAND hip_prefill_fused_iq)
    set_tests_properties(hip_prefill_fused_iq_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 ENVIRONMENT "STRATA_PF_F16=1;STRATA_PF_F16_KQ=1")
    list(APPEND _strata_hip_tests hip_prefill_fused_iq)
    # PR #1368: the MoE input quantized per token and scattered (STRATA_QUANT_GATHER=0) against the per-row gather:
    # the same q8_1 bytes for every MMQ layout (tests/cuda's test; --bench: both at 2K, 8K and 32K tokens)
    add_executable(hip_prefill_mmq_scatter tests/cuda/prefill_mmq_scatter_test.cpp)
    target_include_directories(hip_prefill_mmq_scatter PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_mmq_scatter PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_mmq_scatter COMMAND hip_prefill_mmq_scatter)
    set_tests_properties(hip_prefill_mmq_scatter PROPERTIES TIMEOUT 120)
    list(APPEND _strata_hip_tests hip_prefill_mmq_scatter)
    # PR #1525: swiglu + the down product's q8_1 rows in one kernel (STRATA_PREFILL_SWIGLU_QUANT=1) against the two,
    # byte for byte (tests/cuda's test)
    add_executable(hip_prefill_swiglu_quant tests/cuda/prefill_swiglu_quant_test.cpp)
    target_include_directories(hip_prefill_swiglu_quant PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_swiglu_quant PRIVATE strata_mmq ggml-base strata_hip_runtime)
    add_test(NAME hip_prefill_swiglu_quant COMMAND hip_prefill_swiglu_quant)
    set_tests_properties(hip_prefill_swiglu_quant PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    list(APPEND _strata_hip_tests hip_prefill_swiglu_quant)
    # PR #1525: the MoE combine fused with the hyper-connection write (STRATA_PREFILL_COMBINE_WRITE=1) against the two
    # kernels, bit for bit (tests/cuda's test)
    add_executable(hip_prefill_combine_write tests/cuda/prefill_combine_write_test.cpp)
    target_link_libraries(hip_prefill_combine_write PRIVATE strata_prefill strata_hip_runtime)
    add_test(NAME hip_prefill_combine_write COMMAND hip_prefill_combine_write)
    set_tests_properties(hip_prefill_combine_write PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    list(APPEND _strata_hip_tests hip_prefill_combine_write)

    # Numerical diagnostic: synthetic layouts by default, optional first-expert IQ3_XXS/IQ4_NL check with a pack.
    add_executable(hip_prefill_mmq_parity tests/hip/prefill_mmq_parity.cpp)
    target_include_directories(hip_prefill_mmq_parity PRIVATE "${_strata_gg}/include")
    target_link_libraries(hip_prefill_mmq_parity PRIVATE strata_prefill)
    add_test(NAME hip_prefill_mmq_parity COMMAND hip_prefill_mmq_parity)
    set_tests_properties(hip_prefill_mmq_parity PROPERTIES TIMEOUT 120)
    list(APPEND _strata_hip_tests hip_prefill_mmq_parity)
endmacro()

# HIP gfx906: every CUDA source of the tree (and the ggml-cuda MMQ sources) compiled as HIP
macro(strata_hip_gfx906_sources)
if(STRATA_HIP_GFX906)
  file(GLOB_RECURSE _strata_cu CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro/*.cu
       ${CMAKE_CURRENT_SOURCE_DIR}/tools/vision/*.cu)
  if(_strata_mmq)
    file(GLOB _strata_mmq_cu ${_strata_mmq}/*.cu ${_strata_mmq}/template-instances/*.cu)
    list(APPEND _strata_cu ${_strata_mmq_cu})
    target_compile_definitions(strata_mmq PRIVATE GGML_USE_HIP)
    # #3172: strata_mmq_w4a4 is built from llama.cpp's headers too; without GGML_USE_HIP its common.cuh took the CUDA vendor header
    # (cuda.h, cuda_bf16.h: not found). Here it only reports that W4A4 (RTX 50's FP4 MMA) is not available.
    if(TARGET strata_mmq_w4a4)
      target_compile_definitions(strata_mmq_w4a4 PRIVATE GGML_USE_HIP)
    endif()
    # #3271: three more link errors of a 0.1.44+ gfx906 build, all sources that strata_mmq lists but the glob above never saw:
    # the trellis overlay's quantize.cu and instances (they are generated into the build tree, so the glob missed them: CMake drops a
    # .cu source whose language is not enabled) and moe_f16_kq.cu, which moe_fused_iq.cu calls (a stub here: no gfx11 device code, so
    # f16kq::covers() answers false and the int8 kernels run, as on every other card)
    if(_strata_mmq_tq_overlay)
      set(_strata_tq_hip ${_strata_tq_overlay_dir}/quantize.cu)
      foreach(_t tq2_t tqk6 tqk7 tqp5 tqp6 tqp7)
        list(APPEND _strata_tq_hip ${_strata_tq_overlay_dir}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      list(APPEND _strata_cu ${_strata_tq_hip})
    endif()
    target_sources(strata_mmq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_f16_kq.cu)
  endif()
  set_source_files_properties(${_strata_cu} PROPERTIES LANGUAGE HIP)
endif()

if(STRATA_HIP_GFX906)
  add_executable(gfx906_qsa_query_swizzle_probe bench/gfx906_qsa_query_swizzle_probe.cpp)
  target_link_libraries(gfx906_qsa_query_swizzle_probe PRIVATE strata_kernels ${_strata_gpu_runtime_target})
endif()
endmacro()
