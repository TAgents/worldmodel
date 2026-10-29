from importlib.resources import files

from pglast import parse_plpgsql, parse_sql


def test_packaged_migration_parses_as_postgresql_and_plpgsql():
    for migration in files("world_model.storage").joinpath("migrations").iterdir():
        if migration.name.endswith(".sql"):
            source = migration.read_text()
            assert parse_sql(source)
            if "LANGUAGE plpgsql" in source:
                assert parse_plpgsql(source)
