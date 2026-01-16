# ==============================================================================
# Name..........: ki7mt-ai-lab-core.spec
# Version.......: 1.0.0
# Purpose.......: RPM spec file for KI7MT AI Lab Core package
# Target OS.....: Rocky Linux 9.x / RHEL 9.x (el9)
# Build System..: Simple file-based (no Autotools, no tarball)
# COPR Ready....: Yes (rpkg SCM clone-based build)
# Author........: Greg Beam, KI7MT
# ==============================================================================

Name:           ki7mt-ai-lab-core
Version:        1.0.0
Release:        1%{?dist}
Summary:        Core database schemas for the KI7MT AI Lab WSPR/Solar Project

License:        GPL-3.0-or-later
URL:            https://github.com/KI7MT/ki7mt-ai-lab-core

BuildArch:      noarch

# Runtime dependencies
Requires:       clickhouse-server >= 23.0
Requires:       clickhouse-client >= 23.0
Requires:       bash >= 4.4

Recommends:     clickhouse-common-static

%description
ki7mt-ai-lab-core provides the foundational database schemas for the KI7MT AI
Lab. This package installs optimized ClickHouse schemas designed to handle 10+
billion WSPR (Weak Signal Propagation Reporter) spot records with associated
solar flux indices for propagation correlation analysis.

Schema Components:
  01-wspr_schema.sql    - 15-column immutable raw WSPR spots table
  02-solar_indices.sql  - Solar flux indices (SSN, SFI, Ap/Kp)
  03-solar_silver.sql   - Aggregated solar metrics views
  04-data_mgmt.sql      - Database maintenance procedures
  05-geo_functions.sql  - Maidenhead grid UDFs (placeholder)

Utility Scripts:
  ki7mt-lab-db-init     - Database initialization script
  ki7mt-lab-env         - Environment configuration

Installation Path:
  Schemas: /usr/share/ki7mt/schema/
  Binaries: /usr/bin/

# ==============================================================================
# Prep - Create build directory (no tarball extraction)
# ==============================================================================
%prep
mkdir -p %{_builddir}/%{name}-%{version}

# ==============================================================================
# Build - Nothing to compile
# ==============================================================================
%build
exit 0

# ==============================================================================
# Install - Direct file copy from repository root
# ==============================================================================
%install
mkdir -p %{buildroot}/usr/share/ki7mt/schema/
mkdir -p %{buildroot}/usr/bin/
mkdir -p %{buildroot}/usr/share/doc/%{name}/
mkdir -p %{buildroot}/usr/share/licenses/%{name}/

# Install SQL schemas (copy .in files, strip .in extension)
for f in database/clickhouse/*.sql.in; do
    base=$(basename "$f" .sql.in)
    cp "$f" %{buildroot}/usr/share/ki7mt/schema/${base}.sql
done

# Install utility scripts (copy .in files, strip .in extension)
for f in src/*.in; do
    base=$(basename "$f" .in)
    cp "$f" %{buildroot}/usr/bin/${base}
    chmod +x %{buildroot}/usr/bin/${base}
done

# Install documentation
cp README.md %{buildroot}/usr/share/doc/%{name}/
cp COPYING %{buildroot}/usr/share/licenses/%{name}/

# ==============================================================================
# Post-install script
# ==============================================================================
%post
echo ""
echo "================================================================================"
echo "  KI7MT AI Lab Core - Installation Complete"
echo "================================================================================"
echo ""
echo "  SQL schemas installed to: /usr/share/ki7mt/schema/"
echo "  Utility scripts installed to: /usr/bin/"
echo ""
echo "  QUICK START:"
echo "  1. Ensure ClickHouse server is running:"
echo "     $ sudo systemctl start clickhouse-server"
echo ""
echo "  2. Apply schemas in order:"
echo "     $ for sql in /usr/share/ki7mt/schema/*.sql; do"
echo "         clickhouse-client < \"\$sql\""
echo "       done"
echo ""
echo "================================================================================"
echo ""

# ==============================================================================
# Post-uninstall script
# ==============================================================================
%postun
if [ $1 -eq 0 ]; then
    echo ""
    echo "  KI7MT AI Lab Core removed. Database schemas in ClickHouse NOT dropped."
    echo "  To drop: clickhouse-client --query=\"DROP DATABASE IF EXISTS wspr\""
    echo ""
fi

# ==============================================================================
# Files manifest
# ==============================================================================
%files
%license COPYING
%doc README.md

# Binary scripts
/usr/bin/ki7mt-lab-db-init
/usr/bin/ki7mt-lab-env

# Schema directory and files
%dir /usr/share/ki7mt
%dir /usr/share/ki7mt/schema
/usr/share/ki7mt/schema/01-wspr_schema.sql
/usr/share/ki7mt/schema/02-solar_indices.sql
/usr/share/ki7mt/schema/03-solar_silver.sql
/usr/share/ki7mt/schema/04-data_mgmt.sql
/usr/share/ki7mt/schema/05-geo_functions.sql

# ==============================================================================
# Changelog
# ==============================================================================
%changelog
* Wed Jan 15 2025 Greg Beam <ki7mt@yahoo.com> - 1.0.0-1
- Initial RPM release for Rocky Linux 9 (el9)
- Simple file-based build (no Autotools)
- COPR rpkg SCM integration ready
- 15-column WSPR schema with LowCardinality optimizations
- Solar indices schema for propagation correlation
- Geo functions placeholder for Maidenhead UDFs
- ki7mt-lab-db-init utility for database initialization
- ki7mt-lab-env utility for environment configuration
