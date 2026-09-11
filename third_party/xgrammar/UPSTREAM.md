# XGrammar provenance

- Upstream: https://github.com/mlc-ai/xgrammar
- Revision: `d02ad2b155a5f0c3eaa711950d154218d982ae7e`
- License: Apache-2.0 (`LICENSE`)
- Imported: C++ core and public headers only, with picojson and DLPack headers.

Python bindings, TVM FFI, tests, examples, and unrelated optional dependencies are intentionally
excluded. NInfer owns the explicit source list in this directory.

Local portability patch: `Grammar::FromLark` uses an overload instead of constructing a default
`std::vector<NamedGrammar>` while `NamedGrammar` is still incomplete. This preserves behavior and
allows the public header to compile with libc++ as well as libstdc++.
