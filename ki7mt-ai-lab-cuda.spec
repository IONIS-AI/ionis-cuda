Name:           ki7mt-ai-lab-cuda
Version:        1.1.2
Release:        1%{?dist}
Summary:        CUDA bridge library for KI7MT AI Lab WSPR processing

License:        GPL-3.0-or-later
URL:            https://github.com/KI7MT/ki7mt-ai-lab
VCS:            {{{ git_dir_vcs }}}
Source0:        {{{ git_dir_pack }}}

# Architecture-specific (contains compiled CUDA object)
ExclusiveArch:  x86_64

# Runtime requirements
Requires:       cuda-cudart >= 12.0

# Build requirements (for local builds with nvcc)
# Note: COPR builds use pre-compiled bridge.o from source
BuildRequires:  gcc
BuildRequires:  make

%description
High-performance CUDA bridge library for GPU-accelerated WSPR (Weak Signal
Propagation Reporter) data processing. Provides zero-copy memory management,
async data transfers, and vectorized processing kernels optimized for
NVIDIA RTX 5090 (Blackwell architecture, compute capability 9.0).

Features:
- Zero-copy pinned memory allocation
- Async host-to-device and device-to-host transfers
- CUDA stream management for pipelined processing
- WSPR spot validation kernel
- Vectorized processing kernel (normalize + validate + convert)
- Deduplication kernel
- Callsign sanitization kernel

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}

%description devel
Header files and development documentation for the KI7MT AI Lab CUDA bridge.
Required for building Go applications with CGO that use the CUDA kernels.

%prep
{{{ git_dir_setup_macro }}}

%build
# Pre-compiled bridge.o is included in source
# To rebuild from source (requires CUDA toolkit):
#   nvcc -c -o bridge.o bridge.cu -arch=sm_90 -O3

%install
# Create directories
install -d %{buildroot}%{_libdir}/%{name}
install -d %{buildroot}%{_includedir}/%{name}
install -d %{buildroot}%{_datadir}/%{name}/cuda

# Install pre-compiled CUDA object
install -m 644 src/cuda/bridge.o %{buildroot}%{_libdir}/%{name}/

# Install header file
install -m 644 src/cuda/bridge.h %{buildroot}%{_includedir}/%{name}/

# Install CUDA source files (for reference/rebuilding)
install -m 644 src/cuda/bridge.cu %{buildroot}%{_datadir}/%{name}/cuda/
install -m 644 src/cuda/kernels.cu %{buildroot}%{_datadir}/%{name}/cuda/

%files
%license COPYING
%doc README
%dir %{_libdir}/%{name}
%{_libdir}/%{name}/bridge.o

%files devel
%doc src/cuda/README.md
%dir %{_includedir}/%{name}
%{_includedir}/%{name}/bridge.h
%dir %{_datadir}/%{name}
%dir %{_datadir}/%{name}/cuda
%{_datadir}/%{name}/cuda/*.cu

%changelog
{{{ git_dir_changelog }}}
