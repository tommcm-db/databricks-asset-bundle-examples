# THIS REPOSITORY HAS BEEN MIGRATED TO EMU

**Please navigate to https://github.com/databricks-eng/bundle-examples-internal.**

~

~

~

~

~

~


# bundle-examples-internal

**For public samples, please refer to https://github.com/databricks/bundle-examples.**

## Why internal?

Internal examples are different from public examples because it allows us to experiment, trigger workflows on internal workspaces, and otherwise experiment with automation (e.g. GitHub Actions or Azure DevOps) in ways we do not want to share with customers.

## Usage

Both commands take an `--environment` flag.
If it is not specified, they assume an environment named `default`.

* `databricks bundle validate` -- show final configuration
* `databricks bundle deploy` -- deploy a bundle
