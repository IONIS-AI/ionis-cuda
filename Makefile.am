# Makefile.am for ki7mt-ai-lab-core

# Scripts installed to $(bindir)
bin_SCRIPTS = \
    src/ki7mt-lab-db-init \
    src/ki7mt-lab-env

# SQL schemas installed to $(datadir)/ki7mt/schema
schemadir = $(datadir)/ki7mt/schema
schema_DATA = \
    database/clickhouse/01-wspr_schema.sql \
    database/clickhouse/02-solar_indices.sql \
    database/clickhouse/03-solar_silver.sql \
    database/clickhouse/04-data_mgmt.sql \
    database/clickhouse/05-geo_functions.sql

# Documentation
dist_doc_DATA = README.md CHANGELOG

# Files to include in distribution tarball
EXTRA_DIST = \
    database/clickhouse/01-wspr_schema.sql.in \
    database/clickhouse/02-solar_indices.sql.in \
    database/clickhouse/03-solar_silver.sql.in \
    database/clickhouse/04-data_mgmt.sql.in \
    database/clickhouse/05-geo_functions.sql.in \
    src/ki7mt-lab-db-init.in \
    src/ki7mt-lab-env.in \
    COPYING

# Clean generated files
CLEANFILES = $(bin_SCRIPTS) $(schema_DATA)

# Make scripts executable after install
install-exec-hook:
	chmod +x $(DESTDIR)$(bindir)/ki7mt-lab-db-init
	chmod +x $(DESTDIR)$(bindir)/ki7mt-lab-env
