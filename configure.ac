AC_PREREQ([2.69])
AC_INIT([ki7mt-ai-lab-core], [1.0.0], [ki7mt@yahoo.com])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign -Wall])

# Substitution variables for .in templates
AC_SUBST([PROGRAM], [$PACKAGE_NAME])
AC_SUBST([VERSION], [$PACKAGE_VERSION])
AC_SUBST([COPYRIGHT], ["2024-2026 Greg Beam, KI7MT"])

# Output files
AC_CONFIG_FILES([
    Makefile
    database/clickhouse/01-wspr_schema.sql
    database/clickhouse/02-solar_indices.sql
    database/clickhouse/03-solar_silver.sql
    database/clickhouse/04-data_mgmt.sql
    database/clickhouse/05-geo_functions.sql
    src/ki7mt-lab-db-init
    src/ki7mt-lab-env
])

AC_OUTPUT

AC_MSG_NOTICE([
  ${PACKAGE_NAME} ${PACKAGE_VERSION} configured.

  Type 'make' to build.
  Type 'sudo make install' to install.
])
