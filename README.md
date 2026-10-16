# guild-kernels-cpu (Phase 2 migration preview)

Independent native expert arithmetic: quantized activation conversion,
gate/up SwiGLU rows and down rows; AVX2 / AVX-512 multi-token implementations
and pinned GGML CPU single-token arithmetic. Package `org.guild.kernels.cpu`
exports `external.cpu.native_expert.v1` through SDK 1.1's `CPE1` C contract.
Core retains NUMA worker scheduling, expert residency and memory policy.

This package has executed real Ornith inference with external Qwen and CUDA
router packages together, with bitwise logits, tokens and tested state parity.
The native arithmetic fallback can be compiled out of Core using
`GUILD_BUILTIN_NATIVE_CPU=OFF`. This is **not the complete CPU extraction**:
Core still owns canonical planar Q2 arithmetic, router lookahead, format/pack
interpretation and other CPU operations. Those remain migration work.

## Independent build

Needs an installed Guild Kernel SDK 1.1, CMake 3.24+, C/C++17 compiler and Git.
GGML is fetched independently at immutable llama.cpp revision
`3cf03257f219afbe7334045ff7c6a06ac68c627d`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/guild-sdk
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/build/package"
```

The portable package has an AVX2/FMA/F16C baseline, verified before registration.
SIMD flags apply only to the specialized translation units. Do not apply global
`-march=native` to a distributable plugin. For the explicit AVX-512 GGML arithmetic
configuration used in the Xeon integration measurements, configure with
`-DGUILD_CPU_GGML_AVX512=ON`; its manifest also requires AVX512F/BW/DQ/VL.
Build the variants as different releases/stores; never replace different bytes
under a previously distributed package/version identity.

## Packaging, signing and independent updates

```sh
guild kernel sign build/package --key-id YOUR-KEY-ID --seed-file /secure/release.seed --bundle build/cpu-1.0.0.gkernel
guild kernel trust-key --publisher org.guild-inference --key-id YOUR-KEY-ID --pubkey YOUR-PUBLIC-KEY
guild kernel install build/cpu-1.0.0.gkernel
guild kernel active --json
```

For a new package release, configure another build with
`-DGUILD_PACKAGE_VERSION=1.0.1`, build, test, stage and sign as above. Install
the new `.gkernel`, run inference, then `guild kernel rollback
org.guild.kernels.cpu` to return to the prior release. The integration driver in
Guild verifies both versions execute and that the Core binary digest is unchanged.
Active package leases reject replacement while the package is in use.

The plugin initializes its **own** GGML conversion tables before publishing
callbacks; Core's GGML state is not shared across the DSO. Omitting this produced
a real-model parity failure during extraction, despite synthetic arithmetic tests.

## Attribution and remaining gates

Arithmetic sources were exported from Guild `42018db` with the explicit mapping
in `SOURCE_PROVENANCE.json`. The format header uses SDK's C representation.
`LICENSE` preserves Guild and Strata MIT notices; `third_party/ggml/LICENSE`
preserves GGML attribution. GGML remains private to the plugin (`-Bsymbolic` on
Linux). No STL objects cross its binary boundary.

Linux x86-64 is the tested platform. Windows release packaging and complete
canonical-Q2 execution are not verified. The retained Core copies are migration
references, not evidence that the remaining CPU extraction is finished.
