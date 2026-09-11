cmake_minimum_required(VERSION 3.18)
project(ninfer_xgrammar_vendor LANGUAGES CXX)

add_library(ninfer_xgrammar STATIC
  cpp/compiled_grammar.cc
  cpp/config.cc
  cpp/earley_parser.cc
  cpp/fsm.cc
  cpp/fsm_builder.cc
  cpp/grammar.cc
  cpp/grammar_builder.cc
  cpp/grammar_compiler.cc
  cpp/grammar_functor.cc
  cpp/grammar_matcher.cc
  cpp/grammar_parser.cc
  cpp/grammar_printer.cc
  cpp/json_schema_converter.cc
  cpp/json_schema_converter_ext.cc
  cpp/lark_converter.cc
  cpp/regex_converter.cc
  cpp/structural_tag.cc
  cpp/suffix_automata.cc
  cpp/support/logging.cc
  cpp/support/recursion_guard.cc
  cpp/testing.cc
  cpp/tokenizer_info.cc)

target_include_directories(ninfer_xgrammar
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/picojson)
target_compile_features(ninfer_xgrammar PUBLIC cxx_std_20)
target_compile_definitions(ninfer_xgrammar PRIVATE XGRAMMAR_ENABLE_CPPTRACE=0)
add_library(ninfer::xgrammar ALIAS ninfer_xgrammar)
